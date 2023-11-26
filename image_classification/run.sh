#method_list=(Euler RK2 RK4 RK23 Sym12Async RK12 Dopri5)
method_list=(RK12)
use_ode=True

for method in ${method_list[@]}; do
    python3 train_mem.py --num_epochs 10 --method ${method} --use_ode ${use_ode}
done
