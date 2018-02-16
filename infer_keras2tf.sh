
# --------<TO BE EDIT>----------------------
# setp 3 & 4 & 10
mainPath="/home/phoebehuang/itri/20180214/" #<your folder path>
inferImgPath="cat.jpeg" # you can change to the other local path or an URL
model="inception_v3"  # ("inception_v3","vgg16","vgg19","resnet","mobilenet","xception")
selfDefinedFileName="try" #<your self-defined name for this inference>


# --------<DO NOT EDIT>----------------------
# setp 3
cd ${mainPath}
oriModelPath=${mainPath}"ori_model_meta/" 
IRpath=${mainPath}"gen_pb_json_npy/"
HWIRpath=${mainPath}"json_hw/" 
CodePath=${mainPath}"gen_code/";echo "" >> ${CodePath}__init__.py
genModelPath=${mainPath}"gen_model/"
mkdir "ori_model_meta/" "gen_pb_json_npy/" "gen_code/"  "gen_model/" "json_hw/"

# step 5
python -m mmdnn.conversion.examples.keras.extract_model -n ${model}
#move model to a certain folder
mv imagenet_${model}.h5 ${oriModelPath}; mv imagenet_${model}.json ${oriModelPath}
# step 6
#Convert architecture from Keras to IR
python -m mmdnn.conversion._script.convertToIR -f keras -d ${IRpath}${selfDefinedFileName}_${model} -n ${oriModelPath}imagenet_${model}.json
#Convert model (including architecture and weights) from Keras to IR
python -m mmdnn.conversion._script.convertToIR -f keras -d ${IRpath}${selfDefinedFileName}_${model} -n ${oriModelPath}imagenet_${model}.json -w ${oriModelPath}imagenet_${model}.h5
# step 7
python -m mmdnn.conversion._script.IRToCode -f tensorflow --IRModelPath ${IRpath}${selfDefinedFileName}_${model}.pb --IRWeightPath ${IRpath}${selfDefinedFileName}_${model}.npy --dstModelPath ${CodePath}${selfDefinedFileName}_tensorflow_${model}.py
# step 8
cd ${CodePath}
python -m mmdnn.conversion.examples.tensorflow.imagenet_test -n ${selfDefinedFileName}_tensorflow_${model}.py -w ${IRpath}${selfDefinedFileName}_${model}.npy --dump ${genModelPath}${selfDefinedFileName}_tf_${model}.ckpt
cd ${mainPath}
# step 9
python inference_tf.py -n ${selfDefinedFileName}
# step 10
inferImgPath="cat.jpeg" # you can change to the other local path or an URL
python -c "from try_inference_tf import inference;inference('${inferImgPath}','${model}' )"
