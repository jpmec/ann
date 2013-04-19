Gem::Specification.new do |s|
  s.name        = 'backproprb'
  s.version     = '0.0.1'
  s.date        = '2013-04-20'
  s.summary     = "Backpropagation Neural Network"
  s.description = "A gem for training and running Backprop Neural Networks"
  s.authors     = ["Joshua Petitt"]
  s.email       = 'joshpetitt@jpmec.com'
  s.homepage    = 'https://github.com/jpmec/ann'
  s.files       = Dir.glob('lib/**/*.rb') +
                  Dir.glob('ext/**/*.{h,c}')
  s.extensions  = ['ext/backproprb/extconf.rb']
  s.test_files  = ['test/test_backproprb.rb']
end
