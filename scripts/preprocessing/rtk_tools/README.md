# RTK command-line tools

`cbct_synthesis.py` calls three RTK tools: `rtksimulatedgeometry`,
`rtkforwardprojections` and `rtkfdk`. A compiled RTK provides them. Otherwise,
the three scripts of this folder provide them on top of the `itk-rtk` wheel:

```bash
pip install itk-rtk
python scripts/preprocessing/cbct_synthesis.py --input 2ABA006.mha --outdir sim \
  --rtk-bin-dir scripts/preprocessing/rtk_tools
```
