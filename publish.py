import xarray as xr
import xpublish
from fastapi.middleware.cors import CORSMiddleware


import xarray as xr
from fastapi import APIRouter, Depends
from xpublish.dependencies import get_dataset

file = "/home/user/code/cocofeats/_outputs/dataset1/sub-S0/ses-SE0/sub-S0_ses-SE0_task-T0_acq-A0_run-0.vhdr@SpectrumArrayWelch.nc"
file = "/home/user/code/cocofeats/_outputs/psychostimulants/sub-1_ses-01_task-RESTING_run-01_eeg.vhdr@SpectrumArrayMultitaper5seg.nc"
ds = xr.open_dataset(file, engine="netcdf4")

router = APIRouter()

@router.get("/coords")
def get_all_coords(dataset: xr.Dataset = Depends(get_dataset)):
    coords_dict = {}
    for name, coord in dataset.coords.items():
        coords_dict[name] = coord.values.tolist()
    return coords_dict

@router.get("/data")
def get_data(dataset: xr.Dataset = Depends(get_dataset)):
    return dataset.to_dict()

rest = ds.rest(routers=[router])

rest.app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


rest.serve(host="0.0.0.0", port=9000)
