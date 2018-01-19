In this benchmark, we launch a spike sorting network with a precomputed
template dictionary for the `fitter` block.

You will find the outputs of this benchmark in:
```
~/.spyking-circus-ort/benchmarks/rates-manipulation/sorting
```
If you want to check the generated data then go to:
```
~/.spyking-circus-ort/benchmarks/rates-manipulation/generation
```
If you want to configure the generation then you can modify the content of:
```
~/.spyking-circus-ort/benchmarks/rates-manipulation/configuration
```


To execute this benchmark use the following instructions:

1. Change your current working directory  
`cd ~/circusort/benchmarks/rates_manipulation`
1. Launch IPython  
`$ ipython`
2. Launch  
`In[1]: %run main.py`

You can alternatively use one of the following instructions for step 2.:

- Launch the generation only  
`In[2]: %run main.py --generation`
- Launch the sorting only *(generation must have been executed once
before)*  
`In[3]: %run main.py --sorting`
