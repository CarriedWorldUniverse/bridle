//go:build windows

package subprocess

import (
	"os"
	"os/exec"
)

// SetPgid is a no-op on windows: there is no POSIX process-group
// concept here. WatchCancelGroup degrades to the same single-process
// signaling WatchCancel already does.
func SetPgid(cmd *exec.Cmd) {}

// signalGroup falls back to signaling cmd.Process directly (no process
// groups on windows).
func signalGroup(cmd *exec.Cmd, sig os.Signal) error {
	return cmd.Process.Signal(sig)
}

// killGroup falls back to killing cmd.Process directly (no process
// groups on windows).
func killGroup(cmd *exec.Cmd) error {
	return cmd.Process.Kill()
}
