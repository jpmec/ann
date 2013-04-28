# Tests for test_backproprb module.

require 'test/unit'

require 'pp'
require 'backproprb'
require 'json'




class BackproprbTestCase < Test::Unit::TestCase

  def test_sigmoid
    result = Backproprb::sigmoid(0)

    assert_equal(0.5, result)
  end

  def test_uniform_random_int
    result1 = Backproprb::uniform_random_int
    result2 = Backproprb::uniform_random_int

    assert_not_equal(result1, result2)
  end

end




class BackproprbLayerTestCase < Test::Unit::TestCase

  def setup
    @test_x_count = 1
    @test_y_count = 1

    @sut = Backproprb::Layer.new @test_x_count, @test_y_count
  end

  def test_setup
    puts "test_setup"
    assert_not_nil @sut
  end

  def test_x_count
    puts "test_x_count"
    result = @sut.x_count
    assert_equal @test_x_count, result
  end

  def test_y_count
    assert_equal @test_y_count, @sut.y_count
  end

  def test_W
    w = @sut.w

    assert_equal @test_x_count * @test_y_count, w.flatten.length

    w1 = Array.new(w.flatten.length)
    w1.map! { rand }

    @sut.w = w1

    assert_equal w1, @sut.w.flatten

  end

  def test_W_count
    w_count1 = @sut.w_count
    w_count2 = @sut.x_count * @sut.y_count

    assert_equal w_count1, w_count2
  end

  def test_W_sum
    @sut.randomize 1, 0

    # compare method to ruby calculated sum
    sum1 = @sut.w_sum
    sum2 = @sut.w.flatten.reduce :+

    assert_in_delta sum1, sum2
  end

  def test_W_mean
    @sut.randomize 1, 0

    mean1 = @sut.w_mean
    mean2 = (@sut.w.flatten.reduce :+) / @sut.w.flatten.length

    assert_in_delta mean1, mean2
  end

  def test_W_stddev
    @sut.randomize 1, 0

    value = @sut.w_stddev

    if (1 == @test_x_count) && (1 == @test_y_count)
      assert_equal 0.0, value
    else
      refute_equal 0.0, value
    end

  end

  def test_x
    x = @sut.x

    assert_equal @test_x_count, x.length

    @sut.x = [0.0]

    assert_equal 0.0, @sut.x[0]

    @sut.x = [0.5]

    assert_equal 0.5, @sut.x[0]

    @sut.x = [1.0]

    assert_equal 1.0, @sut.x[0]
  end


  def test_y
    y = @sut.y

    assert_equal @test_y_count, y.length

    @sut.y = [0.0]

    assert_equal 0.0, @sut.y[0]

    @sut.y = [0.5]

    assert_equal 0.5, @sut.y[0]


    @sut.y = [1.0]

    assert_equal 1.0, @sut.y[0]

  end

  def test_g
    g = @sut.g

    assert_equal @test_y_count, g.length

    @sut.g = [0.0]

    assert_equal 0.0, @sut.g[0]

    @sut.g = [0.5]

    assert_equal 0.5, @sut.g[0]


    @sut.g = [1.0]

    assert_equal 1.0, @sut.g[0]
  end

  def test_randomize
    @sut.randomize 1, 0
    w1 = @sut.w

    @sut.randomize 1, 0
    w2 = @sut.w

    refute_equal w1, w2
  end


  def test_identity
    @sut.identity
  end

  def test_activate

    @sut.randomize 1, 0
    @sut.randomize 1, 0

    @sut.x =  [0.0]
    @sut.activate
#    pp @sut.y

    @sut.x =  [0.1]
    @sut.activate
#    pp @sut.y

    @sut.x =  [0.5]
    @sut.activate
#    pp @sut.y

    @sut.x =  [0.9]
    @sut.activate
#    pp @sut.y

    @sut.x =  [1.0]
    @sut.activate
#    pp @sut.y

  end

  def test_prune
    @sut.randomize 1, 0
    w1 = @sut.w

    @sut.prune 10
    w2 = @sut.w

    refute_equal w1, w2
  end

  def test_from_hash_to_hash
    @sut.randomize 1, 0

    x1 = Array.new(@sut.x.length)
    x1.map! { rand }

    w1 = Array.new(@sut.w.length)
    w1.map! { rand }

    y1 = Array.new(@sut.y.length)
    y1.map! { rand }

    h = {
      'x' => x1,
      'w' => w1,
      'y' => y1
    }

    @sut.from_hash h

#    assert_equal h, @sut.to_hash
  end
end



class BackproprbNetworkTestCase < Test::Unit::TestCase

  def setup
    @test_x_size = 1
    @test_y_size = 1
    @test_layers_count = 2

    @sut = Backproprb::Network.new({"x_size" => @test_x_size,
                                    "y_size" => @test_y_size,
                                    "layer_count" => @test_layers_count});
  end

  def test__activate
    y = @sut.activate "a"
#    puts "a : #{y.hex}"

    y = @sut.activate "b"
#    puts "b : #{y.hex}"
  end

  def test__x_size
    x_size = @sut.x_size

    assert_equal @test_x_size, x_size
  end

  def test__y_size
    y_size = @sut.y_size

    assert_equal @test_y_size, y_size
  end

  def test__layers_count
    layers_count = @sut.layers_count

    assert_equal @test_layers_count, layers_count
  end

  def test__layer_get
    layer = @sut.layer_get 0
    refute_nil layer

    layer = @sut.layer_get 1
    refute_nil layer

    layer = @sut.layer_get(-1)
    refute_nil layer
  end

  def test__jitter
    #assert_in_delta 0, @sut.jitter

    @sut.jitter = 0.1

        jitter = @sut.jitter

#        puts "jitter=#{jitter}"

      #assert_in_delta 0.1, jitter
  end

  def test__randomize
    @sut.randomize 1, 0
  end

  def test__reset
    @sut.reset
  end

  def test__prune
    @sut.prune 0.1
  end

  def test__stats
    stats = @sut.stats

    refute_nil stats

    assert_equal @test_x_size, stats.x_size
    assert_equal @test_y_size, stats.y_size
    assert_equal @test_layers_count, stats.layers_count
  end

  def test__to_hash
    h = @sut.to_hash

    refute_nil h
  end

  def test__to_file__from_file
    filename = "#{self.class}_#{__method__}.txt"

    #@sut.randomize 2
    @sut.identity
    @sut.to_file filename

    sut2 = Backproprb::Network.new({"x_size" => @test_x_size,
                                    "y_size" => @test_y_size,
                                    "layer_count" => @test_layers_count})
    sut2.randomize 2, 0

    sut2.from_file filename

    assert_equal @sut.to_hash, sut2.to_hash

  end

  def test__deep_copy

  end


  def teardown
  end

end




class BackproprbTrainingSetTestCase < Test::Unit::TestCase

  def test__new
    training_set = Backproprb::TrainingSet.new ["a", "b", "c"], ["x", "y", "z"]
  end

  def test__to_hash
    training_set = Backproprb::TrainingSet.new ["a", "b", "c"], ["x", "y", "z"]

    hash = training_set.to_hash

    assert_equal 3, hash["count"]
    assert_equal 1, hash["x_size"]
    assert_equal 1, hash["y_size"]

  end

  def test__count
    training_set = Backproprb::TrainingSet.new ["a"], ["x"]

    result = training_set.count
    assert_equal(1, result)


    training_set = Backproprb::TrainingSet.new ["a", "b"], ["x", "y"]

    result = training_set.count
    assert_equal(2, result)


    training_set = Backproprb::TrainingSet.new ["a", "b", "c"], ["x", "y", "z"]

    result = training_set.count
    assert_equal(3, result)
  end

  def test__x_size
    training_set = Backproprb::TrainingSet.new ["a", "b", "c"], ["x", "y", "z"]

    result = training_set.x_size
    assert_equal(1, result)

    training_set = Backproprb::TrainingSet.new ["aa", "bb", "cc"], ["x", "y", "z"]

    result = training_set.x_size
    assert_equal(2, result)
  end

  def test__y_size
    training_set = Backproprb::TrainingSet.new ["a", "b", "c"], ["x", "y", "z"]

    result = training_set.y_size
    assert_equal(1, result)

    training_set = Backproprb::TrainingSet.new ["a", "b", "c"], ["xx", "yy", "zz"]

    result = training_set.y_size
    assert_equal(2, result)
  end


  def test__x_at
    training_set = Backproprb::TrainingSet.new ["a", "b", "c"], ["x", "y", "z"]

    result = training_set.x_at 0
    assert_equal("a", result)

    result = training_set.x_at 1
    assert_equal("b", result)

    result = training_set.x_at 2
    assert_equal("c", result)
  end


  def test__y_at
    training_set = Backproprb::TrainingSet.new ["a", "b", "c"], ["x", "y", "z"]

    result = training_set.y_at 0
    assert_equal("x", result)

    result = training_set.y_at 1
    assert_equal("y", result)

    result = training_set.y_at 2
    assert_equal("z", result)
  end


#  def test__to_file__from_file
#    filename = "#{self.class}_#{__method__}.txt"
#
#    sut1 = Backproprb::TrainingSet.new ["a", "b", "c"], ["x", "y", "z"]
#    sut1.to_file filename
#
#    sut2 = Backproprb::TrainingSet.new nil, nil
#    sut2.from_file filename
#
#    assert_equal sut1.to_hash, sut2.to_hash
#  end
end




class CBackproprbTrainerTestCase  < Test::Unit::TestCase

  def test__to_hash
    @network = Backproprb::Network.new({"x_size"=>1, "y_size"=>1, "layer_count"=>2})
    @network.randomize 2, 0

    @sut = Backproprb::Trainer.new @network

    assert_not_nil @sut.to_hash
  end


  def test__exercise
    @network = Backproprb::Network.new({"x_size"=>1, "y_size"=>1, "layer_count"=>2})
    @network.randomize 2, 0

    @training_set = Backproprb::TrainingSet.new ["a", "b", "c"], ["x", "y", "z"]
    @exercise_stats = Backproprb::ExerciseStats.new
    @sut = Backproprb::Trainer.new @network

    result = @sut.exercise @exercise_stats, @network, @training_set

    assert_not_nil result

    assert 0 < @exercise_stats.exercise_clock_ticks
  end


  def test__teach_pair
    filename = "#{self.class}_#{__method__}.txt"

    @network = Backproprb::Network.new({"x_size"=>1, "y_size"=>1, "layer_count"=>2})
    @network.randomize 1, 0
    @network.from_file filename

    @training_stats = Backproprb::TrainingStats.new
    @sut = Backproprb::Trainer.new @network

    result = @sut.teach_pair @training_stats, @network, "a", "b"

    @network.to_file filename

    assert_not_nil result
    assert_equal 0, result
    assert_equal "b", @network.activate("a")
  end


  def test__train_pair
    filename = "#{self.class}_#{__method__}.txt"

    @network = Backproprb::Network.new({"x_size"=>1, "y_size"=>1, "layer_count"=>2})
    @network.randomize 2, 0
    @network.from_file filename

    @training_stats = Backproprb::TrainingStats.new
    @sut = Backproprb::Trainer.new @network

    @sut.set_to_verbose_io

    result = @sut.train_pair @training_stats, @network, "a", "b"

    assert_not_nil result
    assert_equal 0, result
    assert_equal "b", @network.activate("a")

    @network.to_file filename
  end


  def test__train_set
    filename = "#{self.class}_#{__method__}.txt"

    @network = Backproprb::Network.new({"x_size"=>1, "y_size"=>1, "layer_count"=>2})
    @network.randomize 2, 0
    @network.from_file filename

    @training_set = Backproprb::TrainingSet.new ["a"], ["b"]
    @training_stats = Backproprb::TrainingStats.new
    @sut = Backproprb::Trainer.new @network

    result = @sut.train_set @training_stats, @network, @training_set

    @network.to_file filename

    assert_not_nil result
    assert_equal 0, result
    assert_equal "b", @network.activate("a")
  end


  def test__train_batch
    filename = "#{self.class}_#{__method__}.txt"

    @network = Backproprb::Network.new({"x_size"=>1, "y_size"=>1, "layer_count"=>2})
    @network.randomize 2, 0
    @network.from_file filename

    @training_set = Backproprb::TrainingSet.new ["a"], ["b"]
    @training_stats = Backproprb::TrainingStats.new
    @exercise_stats = Backproprb::ExerciseStats.new
    @sut = Backproprb::Trainer.new @network

    result = @sut.train_batch @training_stats, @exercise_stats, @network, @training_set

    @network.to_file filename

    assert_not_nil result
    assert_equal 0, result
    assert_equal "b", @network.activate("a")
  end


  def test__train
    @network = Backproprb::Network.new({"x_size"=>1, "y_size"=>1, "layer_count"=>2})
    @network.randomize 2, 0

    @training_set = Backproprb::TrainingSet.new ["a"], ["b"]
    @training_stats = Backproprb::TrainingStats.new
    @exercise_stats = Backproprb::ExerciseStats.new
    @sut = Backproprb::Trainer.new @network

    result = @sut.train @training_stats, @exercise_stats, @network, @training_set

    assert_not_nil result
    assert_equal 0, result
    assert_equal "b", @network.activate("a")

    filename = "#{self.class}_#{__method__}.txt"
    @network.to_file filename
  end

end




class CBackproprbEvolverTestCase  < Test::Unit::TestCase

  def test__new
    @sut = Backproprb::Evolver.new

    assert_not_nil @sut
  end


  def test__to_hash
    @sut = Backproprb::Evolver.new

    hash = @sut.to_hash

    assert_not_nil hash
  end


  def test__set_to_default
    @sut = Backproprb::Evolver.new

    @sut.set_to_default

    hash = @sut.to_hash

    assert_not_nil hash
  end


  def test__evolve_caeser_encode
    filename = "#{self.class}_#{__method__}.txt"

    @network = Backproprb::Network.new({"x_size"=>1, "y_size"=>1, "layer_count"=>2})
    @network.randomize 2, 0

    # evolve
    i = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
    o = ["d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "a", "b", "c"]

    @training_set = Backproprb::TrainingSet.new i, o
    @training_stats = Backproprb::TrainingStats.new
    @exercise_stats = Backproprb::ExerciseStats.new
    @trainer = Backproprb::Trainer.new @network
    @evolution_stats = Backproprb::EvolutionStats.new
    @sut = Backproprb::Evolver.new
    @sut.set_to_default

    result = @sut.evolve @evolution_stats, @trainer, @training_stats, @exercise_stats, @network, @training_set

    @network.to_file filename

    assert_not_nil result
    assert_equal 0, result
    assert_equal "d", @network.activate("a")
    assert_equal "e", @network.activate("b")
    assert_equal "f", @network.activate("c")
    assert_equal "g", @network.activate("d")
    assert_equal "h", @network.activate("e")
    assert_equal "i", @network.activate("f")
    assert_equal "j", @network.activate("g")
    assert_equal "k", @network.activate("h")
    assert_equal "l", @network.activate("i")
    assert_equal "m", @network.activate("j")
    assert_equal "n", @network.activate("k")
    assert_equal "o", @network.activate("l")
    assert_equal "p", @network.activate("m")
    assert_equal "q", @network.activate("n")
    assert_equal "r", @network.activate("o")
    assert_equal "s", @network.activate("p")
    assert_equal "t", @network.activate("q")
    assert_equal "u", @network.activate("r")
    assert_equal "v", @network.activate("s")
    assert_equal "w", @network.activate("t")
    assert_equal "x", @network.activate("u")
    assert_equal "y", @network.activate("v")
    assert_equal "z", @network.activate("w")
    assert_equal "a", @network.activate("x")
    assert_equal "b", @network.activate("y")
    assert_equal "c", @network.activate("z")
  end




  def test__evolve_caeser_decode
    filename = "#{self.class}_#{__method__}.txt"

    @network = Backproprb::Network.new({"x_size"=>1, "y_size"=>1, "layer_count"=>2})
    @network.randomize 2, 0

    # evolve
    i = ["d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "a", "b", "c"]
    o = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

    @training_set = Backproprb::TrainingSet.new i, o
    @training_stats = Backproprb::TrainingStats.new
    @exercise_stats = Backproprb::ExerciseStats.new
    @trainer = Backproprb::Trainer.new @network
    @evolution_stats = Backproprb::EvolutionStats.new
    @sut = Backproprb::Evolver.new
    @sut.set_to_default

    result = @sut.evolve @evolution_stats, @trainer, @training_stats, @exercise_stats, @network, @training_set

    @network.to_file filename

    assert_not_nil result
    assert_equal 0, result
    assert_equal "x", @network.activate("a")
    assert_equal "y", @network.activate("b")
    assert_equal "z", @network.activate("c")
    assert_equal "a", @network.activate("d")
    assert_equal "b", @network.activate("e")
    assert_equal "c", @network.activate("f")
    assert_equal "d", @network.activate("g")
    assert_equal "e", @network.activate("h")
    assert_equal "f", @network.activate("i")
    assert_equal "g", @network.activate("j")
    assert_equal "h", @network.activate("k")
    assert_equal "i", @network.activate("l")
    assert_equal "j", @network.activate("m")
    assert_equal "k", @network.activate("n")
    assert_equal "l", @network.activate("o")
    assert_equal "m", @network.activate("p")
    assert_equal "n", @network.activate("q")
    assert_equal "o", @network.activate("r")
    assert_equal "p", @network.activate("s")
    assert_equal "q", @network.activate("t")
    assert_equal "r", @network.activate("u")
    assert_equal "s", @network.activate("v")
    assert_equal "t", @network.activate("w")
    assert_equal "u", @network.activate("x")
    assert_equal "v", @network.activate("y")
    assert_equal "w", @network.activate("z")
  end


  def test__evolve_xor
    filename = "#{self.class}_#{__method__}.txt"

    @network = Backproprb::Network.new({"x_size"=>2, "y_size"=>1, "layer_count"=>2})
    @network.randomize 2, 0

    # evolve
    i = ["00", "01", "10", "11"]
    o = ["0",  "1",  "1",  "0"]

    @training_set = Backproprb::TrainingSet.new i, o
    @training_stats = Backproprb::TrainingStats.new
    @exercise_stats = Backproprb::ExerciseStats.new
    @trainer = Backproprb::Trainer.new @network
    @evolution_stats = Backproprb::EvolutionStats.new
    @sut = Backproprb::Evolver.new
    @sut.set_to_default

    result = @sut.evolve @evolution_stats, @trainer, @training_stats, @exercise_stats, @network, @training_set

    @network.to_file filename

    assert_not_nil result
    assert_equal 0, result
    assert_equal "0", @network.activate("00")
    assert_equal "1", @network.activate("01")
    assert_equal "1", @network.activate("10")
    assert_equal "0", @network.activate("11")
  end


  # Warning this test may take awhile...
  #def test__evolve_tictactoe
  #  filename = "#{self.class}_#{__method__}.txt"
  #
  #  @network = Backproprb::Network.new(9, 9, 2)
  #  @network.randomize 2
  #  @network.from_file filename
  #
  #  # evolve
  #  i = ["         ",
  #       "x        ",
  #       " x       ",
  #       "  x      ",
  #       "   x     ",
  #       "    x    ",
  #       "     x   ",
  #       "      x  ",
  #       "       x ",
  #       "        x"]
  #  o = ["    x    ",
  #       "x   o    ",
  #       " x  o    ",
  #       "  x o    ",
  #       "   xo    ",
  #       "o   x    ",
  #       "    ox   ",
  #       "    o x  ",
  #       "    o  x ",
  #       "    o   x"]
  #
  #  @training_set = Backproprb::TrainingSet.new i, o
  #  @training_stats = Backproprb::TrainingStats.new
  #  @exercise_stats = Backproprb::ExerciseStats.new
  #  @trainer = Backproprb::Trainer.new @network
  #  @evolution_stats = Backproprb::EvolutionStats.new
  #  @sut = Backproprb::Evolver.new
  #  @sut.set_to_default
  #
  #  result = @sut.evolve @evolution_stats, @trainer, @training_stats, @exercise_stats, @network, @training_set
  #
  #  @network.to_file filename
  #
  #  assert_not_nil result
  #  assert_equal 0, result
  #
  #  y = @network.activate("         ")
  #
  #  assert_equal 9, y.length
  #  assert_equal "    x    ", y
  #end


end
