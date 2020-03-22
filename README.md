
# Assignment *number 4*

- name: Joshua McPherson
- student ID: 20687868

## Dependencies
- pytorch
- numpy
- tqdm
- matplotlib

## Running `main.py`

To run `main.py`, place even_mnist.csv into the data directory and type the following command

```sh
python main.py -o result_dir -n 100
```
## Notes
Network will occasionally get stuck in local minima (roughly once every 10 times training it), Learning Rate aneeling is used to avoid this however it still occurs. The network will give a warning if this happens, please re-run main.py if it does so.

Training loss may increase near the end, this is because KLD weight is aneeled so it increases over epochs, the loss plot displays the actual reconstruction loss and KLD (un-weighted), which in testing always converges.