Artificial Neural Networks
==========================

Backprop
--------

A Backpropagation Network library written in C.  There is also a ruby extension wrapper.


Getting started with Ruby extension:

    $ git clone https://github.com/jpmec/ann.git
    $ cd ann/backprop/src/ruby
    $ rake

This should build and install the backproprb gem.


A simple interactive example:

    # start irb
    $ irb

    # require the gem
    > require 'backproprb'

    # create a new network with 1 byte input, 1 byte output and 2 layers (i.e. 1 hidden layer)
    > net = Backproprb::Network.new 1, 1, 2

    # randomize the network with initial seed of 0
    > net.randomize 0

    # convert network to ruby hash
    > net.to_hash
    


