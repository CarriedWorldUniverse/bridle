//go:build !windows

package subprocess

import (
	"os"
	"syscall"
)

// TermSignal is the graceful-termination signal passed to WatchCancel
// by providers that follow the default SIGTERM-then-SIGKILL shape.
func TermSignal() os.Signal { return syscall.SIGTERM }
