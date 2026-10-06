package main

import "core:fmt"
import "core:os"

fail :: proc(args: ..any, location := #caller_location) -> !
{
    fmt.printf("%s:%d: ", location.file_path, location.line)
    fmt.println(..args)
    os.exit(1)
}
