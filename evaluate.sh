# for model in "bllossom"
# do
#     for lr in 3e-5 1e-4 1e-5
#     do
#         for epoch in 1 5 10
#         do
#             python evaluate_models.py --model $model --lr $lr --epoch $epoch
#         done
#     done
# done

for model in "bllossom"
do
    python evaluate_models.py --model $model
done


# python evaluate_models.py --model Llama3.1 --epoch 10 --lr 1e-4
# python evaluate_models.py --model solar --epoch 1 --lr 1e-4