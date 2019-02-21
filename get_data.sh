TRAIN_DATA_PATH='/data/train/'
TEST_DATA_PATH='/data/test/'
train_file_de='train.de'
train_file_en='train.en'
test_file_de='newstest2012.de'
test_file_en='newstest2012.en'

if [ -a $TRAIN_DATA_PATH$train_file_de ]
then
	echo "Downloading German training file"
	wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de
fi

if [ -a $TRAIN_DATA_PATH$train_file_en ]
then
	echo "Downloading English file..."
	wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en
fi
echo "All training files downloaded in /data/train/"
if [ -a $TEST_DATA_PATH$test_file_de ]
then 
	echo "Downloading German 2012 testing file"
	wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2012.de
fi

if [ -a $TEST_DATA_PATH$test_file_en ]
then
	echo "Downloading English 2012 testing file"
	wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2012.en
fi
echo "Downloaded all test data into /data/test"
