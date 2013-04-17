require 'cbackproprb'

module Backproprb
  include CBackproprb

  class Layer < CLayer
    
    def from_hash h
      self.x = h['x']
      self.w = h['w']
      self.y = h['y']
    end
    
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

