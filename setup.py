from distutils.core import setup
setup(
  name = 's-atmech',         
  packages = ['s-atmech'],   
  version = '2.0',      
  license='MIT', 
  long_description = open('README.md').read(),
  long_description_content_type='text/markdown',
  description = 's-atmech is an independent Open Source, Deep Learning python library which implements attention mechanism as a RNN(Recurrent Neural Network) Layer as Encoder-Decoder system. (only supports Bahdanau Attention right now).',  
  author = 'Somyajit Chakraborty(Sam)',               
  author_email = 'somyajitchppr@gmail.com',     
  url = 'https://github.com/Samsomyajit/s-atmech',   
  download_url = 'https://github.com/Samsomyajit/s-atmech/archive/ver1.0.1.tar.gz',   
  keywords = ['Attention Mechanism', 'Natural Language Processing', 'Copy Mechanism', 'Text Summarization', 'Text Processing', 'Sentiment Analysis'],   # Keywords that define your package best
  install_requires=[         
          'numpy',
          'pandas',
          'tensorflow',
          'matplotlib',
          'scikit-learn',
          'jupyter',
          'pillow',
          'nltk',
          'pyYAML',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',     
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
