PATH_TO_WV3_IMAGES=$1
SUBPATH_TO_MULT_IMAGES=$2
SUBPATH_TO_PAN_IMAGES=$3
PATH_TO_OUTPUT=$4

BASE=$PATH_TO_WV3_IMAGES
PATH_MULT=$SUBPATH_TO_MULT_IMAGES
PATH_PAN=$SUBPATH_TO_PAN_IMAGES
PATH_RESULT=$PATH_TO_OUTPUT

WV2_W='-w 0.095 -w 0.7 -w 0.35 -w 1.0 -w 1.0 -w 1.0 -w 1.0 -w 1.0'
WV3_W='-w 0.005 -w 0.142 -w 0.209 -w 0.144 -w 0.234 -w 0.234 -w 0.157 -w 0.116'

for mult_image in $BASE$PATH_MULT/*.TIF; do
    filename=$(basename $mult_image)
    pan_filename=$(echo "$filename" | sed -e 's/M3DS/P3DS/i')
    pan_image=$BASE$PATH_PAN/$pan_filename
    result_image=$PATH_RESULT/$filename
    gdal_pansharpen.py -nodata 0 $WV2_W $pan_image $mult_image $result_image
done


# test_path=/home/sia/kbae_local  # 띄어씌기 있으면 안됨!
# sub2_path=/shell_test2/

# base=$test_path
# sub=$sub2_path

# for i in $base$sub/*.txt; do
#     filename=$(basename $i)
#     test_filename=$(echo "$filename" | sed -e 's/hihi/byebye/i')
#     cp $base$sub$filename $base$sub$test_filename
# done