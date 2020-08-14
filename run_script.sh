

ml Keras/2.2.5-fosscuda-2019a-Python-3.6.8
ml OpenCV/3.4.6-fosscuda-2019a-Python-3.6.8
ml imageio/2.5.0-fosscuda-2019a-Python-3.6.8
ml scikit-image/0.15.0-fosscuda-2019a-Python-3.6.8
ml IPython/7.5.0-fosscuda-2019a-Python-3.6.8

srun    -p gpu \
        --gres=gpu:1 \
        --mail-user=blahaj22@fel.cvut.cz \
        --mail-type=ALL \
        bash -c "python3 exp1-pos-anom.py > results/exp1_results.txt && python3 exp1-neg-anom.py > results/exp1_results_neg.txt"
