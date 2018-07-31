#!/bin/zsh
#download from https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8
#celebA 前処理

unzip img_align_celeba.zip
mkdir 1_50000
cp  {000001..050000}.jpg 1_50000
cd 1_50000
echo "check width and height"
identify -format "%w,%h\n" {000001..000010}.jpg
#178,218

mid=178
#convert 000121.jpg -crop ${mid}x${mid}+$(((178-mid)/2))+$(((218-mid)/2)) -resize 32x32 out.jpg
#display out.jpg

mogrify -crop ${mid}x${mid}+$(((178-mid)/2))+$(((218-mid)/2)) -resize 32x32 {000001..050000}.jpg

cd ..
cp {050001..050019}.jpg 1_50000
cd 1_50000
mogrify -crop ${mid}x${mid}+$(((178-mid)/2))+$(((218-mid)/2)) -resize 32x32 {050001..050019}.jpg
