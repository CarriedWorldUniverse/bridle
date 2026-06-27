//go:build !windows

package codexcli

import (
	"os"
	"syscall"
)

func sigterm() os.Signal { return syscall.SIGTERM }
