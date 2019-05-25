# neural-machine-translation

Encoder-Decoder and Attention-based model for English -> French translation.

Run with [spell](https://spell.run/):

spell command: spell run --machine-type K80 --framework pytorch  "python src/seq2seq.py"

will train the model and save the preprocessed data to `data/eng-fra-processed.txt` (configurable with the parameter `save_processed`)


If the preprocessed data already exists from the previous run, you can directly specify it, and also you can pass in the --save flag if you want to save the model (weights):

spell run --machine-type K80 --framework pytorch 'python src/seq2seq.py --save --sentences_data data/eng-fra-processed.txt'

To predict, run locally:

python src/seq2seq.py --sentences_data data/eng-fra-processed.txt --encoder models/seq2seq_model_encoder_2019-04-2911:02:56.544932 --decoder models/seq2seq_model_decoder_2019-04-2911:02:56.571288
