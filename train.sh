################################################################## Dynamic Object
output=output/dynamic_object
exp=fan
python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.70 --physics_code 16 --use_mpm --mpm_weight 0.1 --mpm_grid_res 32 --vel_start_time 0.0
# python seg.py -m $output/$exp --K 2 --vis
python render.py -m $output/$exp --mode render --skip_train
python metrics.py -m $output/$exp --s test --half_res
python metrics.py -m $output/$exp --s val --half_res

exp=whale
python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.75 --physics_code 16 --use_mpm --mpm_weight 0.1 --mpm_grid_res 256
python render.py -m $output/$exp --mode render --skip_train
python metrics.py -m $output/$exp --s test --half_res
python metrics.py -m $output/$exp --s val --half_res

exp=shark
python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.75 --physics_code 16 --use_mpm --mpm_weight 0.1 --mpm_grid_res 256
python render.py -m $output/$exp --mode render --skip_train
python metrics.py -m $output/$exp --s test --half_res
python metrics.py -m $output/$exp --s val --half_res

exp=telescope
python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.75 --physics_code 16 --use_mpm --mpm_weight 0.1 --mpm_grid_res 256
python render.py -m $output/$exp --mode render --skip_train
python metrics.py -m $output/$exp --s test --half_res
python metrics.py -m $output/$exp --s val --half_res

exp=fallingball
python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.72 --physics_code 16 --use_mpm --mpm_weight 0.1 --mpm_grid_res 256
python render.py -m $output/$exp --mode render --skip_train
python metrics.py -m $output/$exp --s test --half_res
python metrics.py -m $output/$exp --s val --half_res

exp=bat
python train_gui.py -s /path/to/data/ -m $output/$exp --max_time 0.75 --physics_code 16 --use_mpm --mpm_weight 0.1 --mpm_grid_res 256
# python seg.py -m $output/$exp --K 3 --vis
python render.py -m $output/$exp --mode render --skip_train
python metrics.py -m $output/$exp --s test --half_res
python metrics.py -m $output/$exp --s val --half_res
# python render.py -m $output/$exp --mode all --skip_val --skip_test