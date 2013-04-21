require 'cbackproprb'

module Backproprb
  include CBackproprb

  class Layer < CLayer
  end

  class Network < CNetwork
  end

  class Trainer < CTrainer
  end

  class TrainingSet < CTrainingSet
  end

  class ExerciseStats < CExerciseStats
  end

  class TrainingStats < CTrainingStats
  end

  class EvolutionStats < CEvolutionStats
  end

  class Evolver < CEvolver
  end

end
