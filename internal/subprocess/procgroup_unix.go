//go:build !windows

package subprocess

import (
	"os"
	"os/exec"
	"syscall"
)

// SetPgid marks cmd to start as the leader of its own new process group
// (Setpgid) so a WHOLE PROCESS TREE spawned under it can be reaped
// together via signalGroup/killGroup below, instead of only the direct
// child. This is opt-in per spawn call — call it before cmd.Start() —
// so providers that don't need it (claudecode, codexcli, geminicli) are
// completely unaffected; only a caller that both sets this AND uses
// WatchCancelGroup (not the plain WatchCancel) gets group-kill
// semantics.
//
// Rationale (NEX-745 review gate, HIGH): claudesdk's sidecar spawns the
// real `claude` CLI as its own child (a grandchild of bridle). Signaling
// only the sidecar's PID (the pre-fix WatchCancel behavior) leaves that
// grandchild running — orphaned, still holding the auth token, possibly
// still streaming/spending — because a plain SIGTERM to one PID never
// reaches processes it forked.
func SetPgid(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

// signalGroup sends sig to the process GROUP led by cmd's pid (the
// negative-pid kill(2) convention). Requires SetPgid to have been called
// on cmd before Start; otherwise this degrades to signaling only
// cmd.Process (best-effort, not a hard requirement — callers that forget
// SetPgid still get the old single-PID behavior rather than an error).
func signalGroup(cmd *exec.Cmd, sig os.Signal) error {
	s, ok := sig.(syscall.Signal)
	if !ok {
		return cmd.Process.Signal(sig)
	}
	if err := syscall.Kill(-cmd.Process.Pid, s); err != nil {
		// ESRCH: the group is already gone (process exited/reaped between
		// the caller checking and this call) — not an error worth
		// surfacing, mirrors cmd.Process.Signal's existing "no worse than
		// before" contract.
		if err == syscall.ESRCH {
			return nil
		}
		return err
	}
	return nil
}

// killGroup SIGKILLs the process group led by cmd's pid. See signalGroup
// for the SetPgid precondition.
func killGroup(cmd *exec.Cmd) error {
	if err := syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL); err != nil {
		if err == syscall.ESRCH {
			return nil
		}
		return err
	}
	return nil
}
