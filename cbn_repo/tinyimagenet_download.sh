pwd=$(pwd)

echo $pwd
mkdir -p ~/datasets
cd ~/datasets

if [ ! -d tiny-imagenet-200 ]; then
  wget -nc http://cs231n.stanford.edu/tiny-imagenet-200.zip
  unzip -o tiny-imagenet-200.zip
  rm tiny-imagenet-200.zip
fi

cd $pwd
python tiny_imagenet.py