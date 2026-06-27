//go:build windows

package subprocess

import "os"

// Windows doesn't support SIGTERM for subprocess signaling in the same way.
// Kill() is the graceful option available; we use it directly.
func TermSignal() os.Signal { return os.Interrupt }
