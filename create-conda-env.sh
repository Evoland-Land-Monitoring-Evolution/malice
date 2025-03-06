export python_version="3.10"
export name="alise"
if ! [ -z "$1" ]
then
    export name=$1
fi


export target=.../virtualenv/$name
if ! [ -z "$2" ]
then
    export target="$2/$name"
fi

echo "Installing $name in $target ..."

if [ -d "$target" ]; then
   echo "Cleaning previous conda env"
   rm -rf $target
fi

# Create blank virtualenv

conda env create -f mt_ubarn_environment.yml
conda activate alise
which python
python --version

python -m ipykernel install --user --name "$name"
pip install -U setuptools setuptools_scm wheel

# End
#Install torchmuntan be sure that it is downlaoded beforehand

# shellcheck disable=SC2164
cd ../torchmuntan
pip install -e .
# shellcheck disable=SC2164
cd ../openeo-datasets
pip install -e .
# shellcheck disable=SC2164
cd ../mtan_s1s2_classif
pip install -e .
# shellcheck disable=SC2164
cd ../msenge_dataset
pip install -e .
# shellcheck disable=SC2164
cd ../pastis_eo_dataset
pip install -e .
# shellcheck disable=SC2164
cd ../presto
pip install -e .
# shellcheck disable=SC2164
cd ../alise
pip install -e .
# shellcheck disable=SC2164
cd ../modcx
pip install -e .
conda deactivate
