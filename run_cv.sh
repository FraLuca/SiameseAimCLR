date

#### AimCLR NTU-60 xview ####

# Pretext
python main.py pretrain_aimclr --config config/ntu60/pretext/pretext_aimclr_xview.yaml
python main.py pretrain_siam_aimclr --config config/ntu60/pretext/pretext_siam_aimclr_xview_joint.yaml
python main.py pretrain_siam_neg_aimclr --config config/ntu60/pretext/pretext_siam_neg_aimclr_xview.yaml
python main.py pretrain_byol_aimclr --config config/ntu60/pretext/pretext_byol_aimclr_xview.yaml

# Linear_eval
python main.py linear_evaluation --config config/ntu60/linear_eval/linear_eval_aimclr_xview.yaml
python main.py linear_evaluation --config config/ntu60/linear_eval/linear_eval_siam_aimclr_xview.yaml
python main.py linear_evaluation --config config/ntu60/linear_eval/
python main.py linear_evaluation_byol --config config/ntu60/linear_eval/linear_eval_byol_aimclr_xview.yaml

# Ensemble
python ensemble_ntu_cv.py
