//go:build windows

package codexcli

import "os"

func sigterm() os.Signal { return os.Kill }
