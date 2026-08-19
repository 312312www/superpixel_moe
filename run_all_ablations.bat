@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d F:\moe\superpixel_moe_upload

set "PYTHON=F:\anaconda3\envs\fas-superpixel-moe\python.exe"
set "DATA_ROOT=F:\00Dataset\FAS"
set "LANDMARK_MODEL=F:\moe\superpixel_moe_code_20260811\models\face_landmarker.task"
set "RESNET_WEIGHT=C:\Users\ASUS\.cache\torch\hub\checkpoints\resnet50-11ad3fa6.pth"

set "EPOCHS=10"
set "BATCH_SIZE=6"

rem Locked file hashes:
rem ResNet-50 SHA256:
rem 11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca
rem Face Landmarker SHA256:
rem 64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff

if not exist "%PYTHON%" (
    echo ERROR: Python environment not found: %PYTHON%
    exit /b 1
)

if not exist "%DATA_ROOT%\domain-generalization" (
    echo ERROR: Dataset root not found: %DATA_ROOT%
    exit /b 1
)

if not exist "%LANDMARK_MODEL%" (
    echo ERROR: Landmark model not found: %LANDMARK_MODEL%
    exit /b 1
)

if not exist "%RESNET_WEIGHT%" (
    echo ERROR: ResNet-50 weight not found: %RESNET_WEIGHT%
    exit /b 1
)

for %%P in (OCI_M OMI_C OCM_I ICM_O) do (
    for %%E in (A B C D E) do (
        for %%S in (7 17 27) do (
            set "OUT=outputs\ablations\%%P\%%E\seed_%%S"

            if exist "!OUT!\metrics_test.json" (
                echo [SKIP] %%P experiment %%E seed %%S already completed
            ) else (
                if not exist "!OUT!" mkdir "!OUT!"

                echo.
                echo ============================================================
                echo [START] !date! !time!
                echo Protocol=%%P Experiment=%%E Seed=%%S
                echo Output=!OUT!
                echo ============================================================

                "%PYTHON%" -u train_moe.py ^
                  --dataset-root "%DATA_ROOT%" ^
                  --manifest "splits\ablation_rgb_dg_v1\%%P.json" ^
                  --experiment %%E ^
                  --batch-size !BATCH_SIZE! ^
                  --epochs !EPOCHS! ^
                  --image-range "0-1/255" ^
                  --backbone-learning-rate 1e-5 ^
                  --module-learning-rate 1e-4 ^
                  --weight-decay 1e-4 ^
                  --weights-path "%RESNET_WEIGHT%" ^
                  --pretrained ^
                  --train-backbone ^
                  --freeze-batch-norm ^
                  --landmark-model "%LANDMARK_MODEL%" ^
                  --landmark-cache-dir "outputs\landmark_cache" ^
                  --slic-cache-dir "outputs\slic_cache" ^
                  --amp ^
                  --device cuda ^
                  --seed %%S ^
                  --output-dir "!OUT!" ^
                  > "!OUT!\train.log" 2>&1

                set "RC=!ERRORLEVEL!"

                if not "!RC!"=="0" (
                    echo [FAILED] %%P experiment %%E seed %%S
                    echo Exit code: !RC!
                    echo Check: !OUT!\train.log
                    exit /b !RC!
                )

                if not exist "!OUT!\metrics_test.json" (
                    echo [FAILED] Training exited without metrics_test.json
                    echo Check: !OUT!\train.log
                    exit /b 1
                )

                echo [FINISHED] !date! !time!
                echo Protocol=%%P Experiment=%%E Seed=%%S
            )
        )
    )
)

echo.
echo ============================================================
echo All 60 formal ablation runs completed.
echo ============================================================

endlocal