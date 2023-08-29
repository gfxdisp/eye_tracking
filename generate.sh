#!/usr/bin/env bash

path="/mnt/d/Blender/setups_left"
start=104
end=114

for ((i=start; i<end; i++))
do
  printf -v k "%05d" $i
  echo "Estimating $i/$end"
  cmake-build-release-wsl/hdrmfs_eye_tracker -s $path -f folder -p $path/$k -u blender -c 10 -e 10 -k 00 -t 00 -l -h
  echo "Fitting $i/$end"
  cmake-build-release-wsl/hdrmfs_eye_tracker -s $path -f folder -p $path/$k -u blender -c 10 -e 10 -k 00 -t 00 -l -h
done
