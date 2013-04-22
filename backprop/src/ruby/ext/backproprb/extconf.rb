# Loads mkmf which is used to make makefiles for Ruby extensions
require 'mkmf'

$CFLAGS += " -std=c99"

# Do the work
create_makefile('backproprb')
