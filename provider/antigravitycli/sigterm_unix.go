//go:build !windows

package antigravitycli

import (
	"os"
	"syscall"
)

func sigterm() os.Signal { return syscall.SIGTERM }
