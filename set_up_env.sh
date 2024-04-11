#!/usr/bin/sh


export PATH=${PWD}:${PATH}

alias runwarmup='docker run --mount type=bind,src=${PWD}/data/warmup,target=/dataset --mount type=bind,src=${PWD}/submission,target=/submission astrokinetix'
alias runsplid='docker run --mount type=bind,src=${PWD}/data/phase_2,target=/dataset --mount type=bind,src=${PWD}/submission,target=/submission astrokinetix'
alias runphase1='docker run --mount type=bind,src=${PWD}/data/phase_1_v3,target=/dataset --mount type=bind,src=${PWD}/submission,target=/submission astrokinetix'
