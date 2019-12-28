
#types=(xy angle_vortex angle vortex)
types=(angle vortex)

#deltas=(0.34 1.0 0.2)
deltas=(0.2)

#Ls=(12 16 24 32 40 48 56 60)
Ls=(12 16 24 32)
#Ls=(24 32 40 48 56 60)
#Ls=(56 64)
#Ls=(48 56 60)
for type in ${types[*]}
  do
    for delta in ${deltas[*]}
      do
        for L in ${Ls[*]}
          do
            echo "current type: ${type} ; delta: ${delta} and ; L: ${L}"
            data_dir="/home/nghia/generalized_xy/delta${delta}/hdf5/Delta${delta}/L${L}"
            model_dir="/home/nghia/generalized_xy/codes/model/${type}/Delta${delta}/L${L}"
            #echo "python post_process_on_test.py ${type} ${data_dir} ${model_dir}"
            python post_process_on_test.py ${type} ${data_dir} ${model_dir}
        done
    done
done
