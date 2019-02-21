if [ -a /data/train/train.de ]
then
	echo "Downloading German training file"
	wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de
fi

if [ -a /data/train/train.en ]
then
	echo "Downloading English file..."
	wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en
fi
echo "All training files downloaded in /data/train/"
if [ -a /data/test/newstest2012.de ]
then 
	echo "Downloading German 2012 testing file"
	wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2012.de
fi

if [ -a /data/test/newstest2012.en ]
then
	echo "Downloading English 2012 testing file"
	wget https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2012.en
fi
echo "Downloaded all test data into /data/test"
