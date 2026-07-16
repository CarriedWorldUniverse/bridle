package wset

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	bridle "github.com/CarriedWorldUniverse/bridle"
)

// Manager is wset v2: the two-layer budget (agora-spec-context-curation §3).
//
// Layer 1 (RESIDENT): the live copy of each artifact key held in context,
// under a token budget with LRU demotion, hot-set immunity, and hysteresis.
//
// Layer 2 (TRACKED): demoted keys stay in an in-memory ledger — content,
// disk hash, last-touch — and come BACK on demand:
//
//  1. tracked copy valid (hash matches disk) → un-stub the original bytes at
//     the original position (free; re-aligns the pre-demotion prefix);
//  2. stale but disk-backed → the HARNESS reads the disk and appends fresh
//     content to the triggering tool result (no model step burned; the stub
//     never bounces work back to the model that the harness can do itself);
//  3. no disk ground truth → the tracked copy is served with a provenance
//     marker — it is the only truth there is.
//
// Partial re-admission: when the trigger carries a locus (an edit site), only
// the span around it re-enters (SpanIndexer seam; line-window fallback; files
// under PartialThreshold always re-enter whole).
//
// Eviction pressure stays real: non-keyed results (command output, listings)
// age out past KeepOthers exactly as in v1.
type Manager struct {
	cfg ManagerConfig

	mu   sync.Mutex
	keys map[string]*keyState
	step int
}

// WriteSpec names a full-content write tool's key and content arguments.
type WriteSpec struct{ KeyArg, ContentArg string }

// SpanIndexer resolves the relevant span of a file for a locus — the seam
// where a symbol index (codemap) or LSP plugs in. ok=false → the caller uses
// its line-window fallback.
type SpanIndexer interface {
	SpanFor(rel string, line int) (span, label string, ok bool)
}

// ManagerConfig tunes the two-layer policy. Use DefaultManagerConfig.
type ManagerConfig struct {
	// Root is the sandbox root for keyed paths. When set it enables disk
	// hashing (staleness), harness reads (re-admission source 2), and the
	// post-command modification sweep. Empty = tracked copies are always
	// servable (source 3) and never invalidated.
	Root string

	ReadTools  map[string]string    // tool -> key arg (full-content reads)
	WriteTools map[string]WriteSpec // tool -> key+content args (full-content writes)
	EditTools  map[string]string    // tool -> key arg (mutations WITHOUT full content)

	ContextWindowTokens int     // model context window (tokens)
	Budget              float64 // resident-layer share of the window (0.25)
	EvictTo             float64 // hysteresis floor, as a share of the budget (0.70)
	HotSteps            int     // keys touched within N steps are demotion-immune
	KeepOthers          int     // non-keyed results kept verbatim
	MaxRetainBytes      int     // per-item cap on retained content
	TrackedMaxKeys      int     // ledger entry cap (metadata+content, off-context)
	PartialThreshold    int     // files <= this many bytes always re-enter whole
	ReadmitOnMention    bool    // command output naming a demoted key re-admits it

	Spans SpanIndexer // optional; nil = line-window fallback
}

// DefaultManagerConfig mirrors the spec's defaults for a 128k-token window.
func DefaultManagerConfig(root string) ManagerConfig {
	return ManagerConfig{
		Root:                root,
		ReadTools:           map[string]string{"read_file": "path", "Read": "file_path"},
		WriteTools:          map[string]WriteSpec{"write_file": {"path", "content"}, "Write": {"file_path", "content"}},
		EditTools:           map[string]string{"edit_file": "path", "Edit": "file_path", "apply_patch": "path"},
		ContextWindowTokens: 128000,
		Budget:              0.25,
		EvictTo:             0.70,
		HotSteps:            3,
		KeepOthers:          2,
		MaxRetainBytes:      64 << 10,
		TrackedMaxKeys:      1024,
		PartialThreshold:    4096,
		ReadmitOnMention:    true,
	}
}

// keyState is one ledger entry: the tracked layer for one artifact key.
type keyState struct {
	liveID    string // ToolCallID whose message carries the live copy ("" = none)
	liveArgs  bool   // live copy lives in assistant write args, not a tool result
	demoted   bool   // liveID's message is currently stubbed
	stale     bool   // disk moved on from `content`
	content   string // last-known-good content (the tracked copy)
	label     string // partial-span label, "" = whole
	diskHash  string // sha256 of content when captured (Root-backed keys)
	lastTouch int
	readmit   bool // queued: un-stub at next assembly
}

// AttachManager registers the v2 policy and returns the manager + detach.
func AttachManager(h *bridle.Harness, cfg ManagerConfig) (*Manager, func()) {
	m := &Manager{cfg: cfg, keys: map[string]*keyState{}}
	ids := []bridle.HookID{
		h.RegisterBeforeModelCall(m.beforeModelCall),
		h.RegisterAfterToolCall(m.afterToolCall),
	}
	return m, func() {
		for _, id := range ids {
			h.UnregisterHook(id)
		}
	}
}

const (
	stubDemoted  = "[working set: %s demoted (untouched %d steps, ~%d tokens) — tracked; touch it or re-read to restore]"
	stubStale    = "[working set: %s modified since this read — refreshed content arrives with the next touch, or re-read]"
	refreshWhole = "\n\n[working set refresh — current content of %s]:\n%s"
	refreshPart  = "\n\n[working set refresh — %s %s; rest tracked, re-read for full content]:\n%s"
	writeSupers  = "[superseded by a later write/read of %s; %d chars]"
)

func hashStr(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

func (m *Manager) diskContent(key string) (string, bool) {
	if m.cfg.Root == "" {
		return "", false
	}
	p := filepath.Join(m.cfg.Root, filepath.Clean("/"+key)) // confine to Root
	b, err := os.ReadFile(p)
	if err != nil {
		return "", false
	}
	return string(b), true
}

// freshOnDisk reports whether the tracked copy still matches the disk.
// Keys without disk backing are always "fresh" (source 3: only truth there is).
func (m *Manager) freshOnDisk(ks *keyState, key string) bool {
	cur, ok := m.diskContent(key)
	if !ok {
		return m.cfg.Root == "" // no root: servable; root but unreadable: not
	}
	return hashStr(cur) == ks.diskHash
}

func (m *Manager) touch(key string) *keyState {
	ks := m.keys[key]
	if ks == nil {
		ks = &keyState{}
		m.keys[key] = ks
	}
	ks.lastTouch = m.step
	return ks
}

// ---- afterToolCall: observe, refresh, and deliver where the model looks ----

func (m *Manager) afterToolCall(_ context.Context, in bridle.AfterToolCallCtx) (bridle.AfterToolCallCtx, bridle.HookAction, error) {
	if in.Result.Err != "" {
		return in, bridle.HookContinue, nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.step = in.Step

	argStr := func(arg string) string {
		var args map[string]json.RawMessage
		json.Unmarshal(in.Call.Args, &args)
		var v string
		json.Unmarshal(args[arg], &v)
		return v
	}
	resultStr := func() string {
		var s string
		if json.Unmarshal(in.Result.Result, &s) != nil {
			s = string(in.Result.Result)
		}
		return s
	}

	if keyArg, ok := m.cfg.ReadTools[in.Call.Name]; ok {
		if key := argStr(keyArg); key != "" {
			ks := m.touch(key)
			ks.content = resultStr()
			ks.liveID, ks.liveArgs = in.Call.ID, false
			ks.demoted, ks.stale, ks.readmit, ks.label = false, false, false, ""
			ks.diskHash = hashStr(ks.content)
		}
		return in, bridle.HookContinue, nil
	}

	if spec, ok := m.cfg.WriteTools[in.Call.Name]; ok {
		if key := argStr(spec.KeyArg); key != "" {
			ks := m.touch(key)
			ks.content = argStr(spec.ContentArg)
			ks.liveID, ks.liveArgs = in.Call.ID, true
			ks.demoted, ks.stale, ks.readmit, ks.label = false, false, false, ""
			ks.diskHash = hashStr(ks.content)
		}
		return in, bridle.HookContinue, nil
	}

	if keyArg, ok := m.cfg.EditTools[in.Call.Name]; ok {
		key := argStr(keyArg)
		if key == "" {
			return in, bridle.HookContinue, nil
		}
		ks := m.touch(key)
		fresh, onDisk := m.diskContent(key)
		if !onDisk {
			// cannot see the result of the edit: the old copy is no longer
			// truth and nothing can refresh it — invalidate (correctness rule)
			ks.stale = true
			return in, bridle.HookContinue, nil
		}
		span, label := m.spanAround(key, ks.content, fresh)
		out := resultStr()
		if label == "" {
			out += fmt.Sprintf(refreshWhole, key, span)
		} else {
			out += fmt.Sprintf(refreshPart, key, label, span)
		}
		b, _ := json.Marshal(out)
		in.Result.Result = b
		// this result now carries the live (possibly partial) copy; the prior
		// copy is superseded at the next assembly
		ks.content, ks.diskHash = fresh, hashStr(fresh)
		ks.liveID, ks.liveArgs, ks.label = in.Call.ID, false, label
		ks.demoted, ks.stale, ks.readmit = false, false, false
		return in, bridle.HookContinue, nil
	}

	// other tools (run_command etc.): sweep for modifications + mentions
	res := resultStr()
	for key, ks := range m.keys {
		if m.cfg.Root != "" && ks.content != "" && !ks.stale {
			if cur, ok := m.diskContent(key); ok && hashStr(cur) != ks.diskHash {
				ks.stale = true // the assembly stubs the resident copy
			}
		}
		if !m.cfg.ReadmitOnMention || !ks.demoted || !strings.Contains(res, key) {
			continue
		}
		ks.lastTouch = m.step
		if !ks.stale && m.freshOnDisk(ks, key) {
			ks.readmit = true // source 1: un-stub at next assembly
			continue
		}
		if fresh, ok := m.diskContent(key); ok { // source 2: harness read, deliver here
			res += fmt.Sprintf(refreshWhole, key, m.capped(fresh))
			b, _ := json.Marshal(res)
			in.Result.Result = b
			ks.content, ks.diskHash = fresh, hashStr(fresh)
			ks.liveID, ks.liveArgs, ks.label = in.Call.ID, false, ""
			ks.demoted, ks.stale, ks.readmit = false, false, false
		}
	}
	m.pruneTracked()
	return in, bridle.HookContinue, nil
}

// spanAround picks what re-enters after an edit: the whole file when small,
// else the SpanIndexer's span for the first changed line, else a line window.
func (m *Manager) spanAround(key, old, fresh string) (span, label string) {
	if len(fresh) <= m.cfg.PartialThreshold {
		return m.capped(fresh), ""
	}
	line := firstChangedLine(old, fresh)
	if m.cfg.Spans != nil {
		if s, l, ok := m.cfg.Spans.SpanFor(key, line); ok {
			return m.capped(s), l
		}
	}
	lines := strings.Split(fresh, "\n")
	lo, hi := line-6, line+6
	if lo < 0 {
		lo = 0
	}
	if hi > len(lines) {
		hi = len(lines)
	}
	return m.capped(strings.Join(lines[lo:hi], "\n")), fmt.Sprintf("L%d–%d", lo+1, hi)
}

func firstChangedLine(old, fresh string) int {
	a, b := strings.Split(old, "\n"), strings.Split(fresh, "\n")
	for i := 0; i < len(a) && i < len(b); i++ {
		if a[i] != b[i] {
			return i
		}
	}
	if len(a) < len(b) {
		return len(a)
	}
	return 0
}

func (m *Manager) capped(s string) string {
	return truncateRetained(s, m.cfg.MaxRetainBytes)
}

// pruneTracked bounds the ledger: drop the least-recently-touched DEMOTED
// entries past the cap (the thread/disk still has the bytes; the key just
// reverts to plain re-read).
func (m *Manager) pruneTracked() {
	if m.cfg.TrackedMaxKeys <= 0 || len(m.keys) <= m.cfg.TrackedMaxKeys {
		return
	}
	type kv struct {
		key string
		ks  *keyState
	}
	var demoted []kv
	for k, ks := range m.keys {
		if ks.demoted {
			demoted = append(demoted, kv{k, ks})
		}
	}
	sort.Slice(demoted, func(i, j int) bool { return demoted[i].ks.lastTouch < demoted[j].ks.lastTouch })
	for _, d := range demoted {
		if len(m.keys) <= m.cfg.TrackedMaxKeys {
			break
		}
		delete(m.keys, d.key)
	}
}

// ---- beforeModelCall: the assembly projection ----

func (m *Manager) beforeModelCall(_ context.Context, in bridle.BeforeModelCallCtx) (bridle.BeforeModelCallCtx, bridle.HookAction, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if in.Step > m.step {
		m.step = in.Step
	}
	msgs := in.Request.Messages

	// index all keyed CARRIERS (read results + write args) in message order,
	// plus call-id lookups
	type carrier struct {
		id      string
		pos     int // message index (results: the result msg; writes: the assistant msg)
		isArgs  bool
		name    string
		key     string
		content string
	}
	var carriers []carrier
	resultPos := map[string]int{}
	callKey := map[string]string{}
	pendingRead := map[string]string{} // read call id -> key (result seen later)
	for i, msg := range msgs {
		for _, tc := range msg.ToolCalls {
			var args map[string]json.RawMessage
			json.Unmarshal(tc.Args, &args)
			pick := func(arg string) string {
				var v string
				json.Unmarshal(args[arg], &v)
				return v
			}
			if a, ok := m.cfg.ReadTools[tc.Name]; ok {
				if k := pick(a); k != "" {
					pendingRead[tc.ID] = k
					callKey[tc.ID] = k
				}
			} else if s, ok := m.cfg.WriteTools[tc.Name]; ok {
				if k := pick(s.KeyArg); k != "" {
					callKey[tc.ID] = k
					carriers = append(carriers, carrier{tc.ID, i, true, tc.Name, k, pick(s.ContentArg)})
				}
			}
		}
		if msg.Role == "tool_result" {
			resultPos[msg.ToolCallID] = i
			if k := pendingRead[msg.ToolCallID]; k != "" {
				carriers = append(carriers, carrier{msg.ToolCallID, i, false, "", k, msg.Content})
			}
		}
	}

	// reconcile: adopt keys the ledger has never seen (history predating the
	// attach, session resume) — newest carrier becomes the live copy
	newest := map[string]carrier{}
	for _, c := range carriers {
		newest[c.key] = c // carriers are in message order; last wins
	}
	for key, c := range newest {
		if m.keys[key] != nil || stubbed(c.content) || strings.HasPrefix(c.content, "[working set:") || strings.HasPrefix(c.content, "[superseded by a later") {
			continue
		}
		m.keys[key] = &keyState{
			liveID: c.id, liveArgs: c.isArgs,
			content: c.content, diskHash: hashStr(c.content), lastTouch: m.step,
		}
	}

	// 1. supersession: every keyed carrier that is NOT the live copy is
	// stubbed (results) or rewritten (write args)
	for _, c := range carriers {
		ks := m.keys[c.key]
		isLive := ks != nil && ks.liveID == c.id && ks.liveArgs == c.isArgs
		if isLive {
			continue
		}
		if c.isArgs {
			spec := m.cfg.WriteTools[msgs[c.pos].ToolCalls[indexOfCall(msgs[c.pos].ToolCalls, c.id)].Name]
			m.rewriteWriteArgs(&msgs[c.pos], c.id, spec, c.key)
			continue
		}
		if !stubbed(msgs[c.pos].Content) && !strings.HasPrefix(msgs[c.pos].Content, "[working set:") {
			msgs[c.pos].Content = fmt.Sprintf(stubSuperseded, "read", c.key, len(msgs[c.pos].Content))
		}
	}

	// 2. staleness + queued re-admissions on live copies
	for key, ks := range m.keys {
		if ks.liveID == "" {
			continue
		}
		pos, ok := resultPos[ks.liveID]
		if ks.liveArgs || !ok {
			continue
		}
		switch {
		case ks.stale && !ks.demoted:
			msgs[pos].Content = fmt.Sprintf(stubStale, key)
			ks.demoted = true
		case ks.readmit && ks.demoted:
			msgs[pos].Content = m.capped(ks.content) // source 1: original bytes
			ks.demoted, ks.readmit = false, false
			ks.lastTouch = m.step
		}
	}

	// 3. resident budget: demote LRU past the hysteresis floor
	budget := int(float64(m.cfg.ContextWindowTokens) * m.cfg.Budget)
	resident := 0
	type live struct {
		key string
		ks  *keyState
		pos int
	}
	var candidates []live
	for key, ks := range m.keys {
		if ks.liveID == "" || ks.demoted || ks.liveArgs {
			continue
		}
		pos, ok := resultPos[ks.liveID]
		if !ok {
			continue
		}
		msgs[pos].Content = m.capped(msgs[pos].Content)
		resident += len(msgs[pos].Content) / 4
		candidates = append(candidates, live{key, ks, pos})
	}
	if budget > 0 && resident > budget {
		floor := int(float64(budget) * m.cfg.EvictTo)
		sort.Slice(candidates, func(i, j int) bool { return candidates[i].ks.lastTouch < candidates[j].ks.lastTouch })
		for _, c := range candidates {
			if resident <= floor {
				break
			}
			if m.step-c.ks.lastTouch <= m.cfg.HotSteps {
				continue // hot set is immune
			}
			tok := len(msgs[c.pos].Content) / 4
			c.ks.content = msgs[c.pos].Content // capture the tracked copy
			msgs[c.pos].Content = fmt.Sprintf(stubDemoted, c.key, m.step-c.ks.lastTouch, tok)
			c.ks.demoted = true
			resident -= tok
		}
	}

	// 4. non-keyed results: keep the newest KeepOthers, stub older — but a
	// result carrying a live copy (edit/command refresh) is a working-set
	// carrier, not noise
	liveIDs := map[string]bool{}
	for _, ks := range m.keys {
		if ks.liveID != "" && !ks.liveArgs && !ks.demoted {
			liveIDs[ks.liveID] = true
		}
	}
	var others []int
	for i, msg := range msgs {
		if msg.Role != "tool_result" || stubbed(msg.Content) || strings.HasPrefix(msg.Content, "[working set:") {
			continue
		}
		if callKey[msg.ToolCallID] == "" && !liveIDs[msg.ToolCallID] {
			others = append(others, i)
		}
	}
	for n := 0; n < len(others)-m.cfg.KeepOthers; n++ {
		i := others[n]
		msgs[i].Content = fmt.Sprintf(stubEvicted, len(msgs[i].Content))
	}
	start := len(others) - m.cfg.KeepOthers
	if start < 0 {
		start = 0
	}
	for _, i := range others[start:] {
		msgs[i].Content = m.capped(msgs[i].Content)
	}
	return in, bridle.HookContinue, nil
}

func indexOfCall(tcs []bridle.ToolInvocation, id string) int {
	for i := range tcs {
		if tcs[i].ID == id {
			return i
		}
	}
	return 0
}

// rewriteWriteArgs replaces a superseded write's content argument in the
// assistant tool_use block (structure and id preserved — provider pairing).
func (m *Manager) rewriteWriteArgs(msg *bridle.ProviderMessage, callID string, spec WriteSpec, key string) {
	for ti := range msg.ToolCalls {
		if msg.ToolCalls[ti].ID != callID {
			continue
		}
		var args map[string]json.RawMessage
		if json.Unmarshal(msg.ToolCalls[ti].Args, &args) != nil {
			return
		}
		var cur string
		json.Unmarshal(args[spec.ContentArg], &cur)
		if strings.HasPrefix(cur, "[superseded by a later") {
			return // idempotent
		}
		marker, _ := json.Marshal(fmt.Sprintf(writeSupers, key, len(cur)))
		args[spec.ContentArg] = marker
		b, _ := json.Marshal(args)
		msg.ToolCalls[ti].Args = b
		return
	}
}
