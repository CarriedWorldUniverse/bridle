//go:build windows

package antigravitycli

import "os"

func sigterm() os.Signal { return os.Interrupt }
