mkdir data
cd data
mkdir train valid test
curl -O http://images.cocodataset.org/zips/test2017.zip
curl -O http://images.cocodataset.org/zips/val2017.zip
tar -xf test2017.zip -C train --strip-components=1
tar -xf val2017.zip -C valid --strip-components=1
