// GPU catalogue. Specs: vendor datasheets, dense BF16/FP16 TFLOPS (no sparsity), HBM/GDDR bandwidth TB/s, VRAM GB, TDP W.
// rate = $/GPU-hr: our invoice where we have one, else a public on-demand list price.
// RP RunPod Secure Cloud (13 Sep 2026), LB Lambda, VI Vast index. fleet = our supply.
// Apple: FP32 rate (FP16 runs at the same rate), cores x 128 lanes x 2 x clock; not published by Apple.
// ftq: cal calibrated | x spec-estimate | no (not fine-tune quoted until a calibration run).
// nvbw: practical GPU-to-GPU GB/s one way over NVLink or equivalent (about 80% of spec); pcie: 50 for PCIe 5.0 cards (default 25)
const RP="RunPod list", LB="Lambda list", VI="Vast index", DS="datasheet";
const GPUS = {
 // ---- NVIDIA data centre ----
 b300:      {nvbw:720,nvlink:1,fp8:1,name:"B300 288GB",sn:"B300",v:"NVIDIA",tier:"dc",tf:2250,bw:8.0,vram:288,tdp:1100,rate:7.89,rsrc:RP,fleet:false,ftq:"no",mkt:{},maxN:8},
 b200:      {nvbw:720,nvlink:1,fp8:1,name:"B200 180GB",sn:"B200",v:"NVIDIA",tier:"dc",tf:2250,bw:8.0,vram:180,tdp:1000,rate:6.79,rsrc:RP,fleet:false,ftq:"no",mkt:{},maxN:8},
 gb200:     {nvbw:720,nvlink:1,fp8:1,name:"GB200 192GB",sn:"GB200",v:"NVIDIA",tier:"dc",tf:2500,bw:8.0,vram:192,tdp:1200,rate:null,rsrc:"no rate on file",fleet:false,ftq:"no",mkt:{},maxN:8},
 h200:      {nvbw:360,nvlink:1,fp8:1,name:"H200 141GB",sn:"H200",v:"NVIDIA",tier:"dc",tf:989,bw:4.8,vram:141,tdp:700,rate:4.59,rsrc:RP,fleet:false,ftq:"cal",mkt:{},maxN:8},
 h100:      {nvbw:360,nvlink:1,fp8:1,name:"H100 80GB SXM",sn:"H100",v:"NVIDIA",tier:"dc",tf:989,bw:3.35,vram:80,tdp:700,rate:2.60,rsrc:"Gcore PAYG, invoiced",fleet:true,ftq:"cal",mkt:{Lambda:3.99,RunPod:3.49,Vast:1.60},maxN:8},
 h100nvl:   {nvbw:240,pcie:50,nvpair:1,fp8:1,name:"H100 NVL 94GB",sn:"H100 NVL",v:"NVIDIA",tier:"dc",tf:835,bw:3.9,vram:94,tdp:400,rate:3.19,rsrc:RP,fleet:false,ftq:"x",mkt:{},maxN:8},
 h100pcie:  {pcie:50,fp8:1,name:"H100 PCIe 80GB",sn:"H100 PCIe",v:"NVIDIA",tier:"dc",tf:756,bw:2.0,vram:80,tdp:350,rate:2.89,rsrc:RP,fleet:false,ftq:"x",mkt:{},maxN:8},
 gh200:     {fp8:1,name:"GH200 96GB",sn:"GH200",v:"NVIDIA",tier:"dc",tf:989,bw:4.0,vram:96,tdp:700,rate:2.29,rsrc:LB,fleet:false,ftq:"x",mkt:{},maxN:1},
 a100:      {nvbw:240,nvlink:1,name:"A100 80GB SXM",sn:"A100",v:"NVIDIA",tier:"dc",tf:312,bw:2.039,vram:80,tdp:400,rate:1.425,rsrc:"Gcore committed, invoiced",fleet:true,ftq:"cal",mkt:{Lambda:2.79,RunPod:1.59,Vast:1.09},maxN:8},
 a100pcie:  {name:"A100 PCIe 80GB",sn:"A100 PCIe",v:"NVIDIA",tier:"dc",tf:312,bw:1.935,vram:80,tdp:300,rate:1.59,rsrc:RP,fleet:false,ftq:"x",mkt:{},maxN:8},
 "a100-40": {name:"A100 40GB",sn:"A100 40GB",v:"NVIDIA",tier:"dc",tf:312,bw:1.555,vram:40,tdp:400,rate:1.99,rsrc:LB,fleet:false,ftq:"x",mkt:{},maxN:8},
 l40s:      {fp8:1,name:"L40S 48GB",sn:"L40S",v:"NVIDIA",tier:"dc",tf:362,bw:0.864,vram:48,tdp:350,rate:1.09,rsrc:RP,fleet:false,ftq:"x",mkt:{},maxN:8},
 l40:       {fp8:1,name:"L40 48GB",sn:"L40",v:"NVIDIA",tier:"dc",tf:181,bw:0.864,vram:48,tdp:300,rate:0.82,rsrc:RP,fleet:false,ftq:"x",mkt:{},maxN:8},
 a40:       {name:"A40 48GB",sn:"A40",v:"NVIDIA",tier:"dc",tf:149.7,bw:0.696,vram:48,tdp:300,rate:0.49,rsrc:RP,fleet:false,ftq:"x",mkt:{},maxN:8},
 a10:       {name:"A10 24GB",sn:"A10",v:"NVIDIA",tier:"dc",tf:125,bw:0.6,vram:24,tdp:150,rate:1.29,rsrc:LB,fleet:false,ftq:"x",mkt:{},maxN:8},
 l4:        {fp8:1,name:"L4 24GB",sn:"L4",v:"NVIDIA",tier:"dc",tf:121,bw:0.3,vram:24,tdp:72,rate:0.49,rsrc:RP,fleet:false,ftq:"x",mkt:{},maxN:8},
 t4:        {name:"T4 16GB",sn:"T4",v:"NVIDIA",tier:"dc",tf:65,bw:0.32,vram:16,tdp:70,rate:null,rsrc:"no rate on file",fleet:false,ftq:"x",mkt:{},maxN:8},
 v100:      {nvbw:120,nvlink:1,name:"V100 16GB",sn:"V100",v:"NVIDIA",tier:"dc",tf:125,bw:0.9,vram:16,tdp:300,rate:0.79,rsrc:LB+" (our GCP rate not on file)",fleet:true,ftq:"x",mkt:{},maxN:4},
 "v100-32":   {nvbw:120,nvlink:1,name:"V100 32GB",sn:"V100 32GB",v:"NVIDIA",tier:"dc",tf:125,bw:0.9,vram:32,tdp:300,rate:null,rsrc:"no rate on file",fleet:false,ftq:"x",mkt:{},maxN:8},
 // ---- NVIDIA workstation / consumer ----
 rtxpro6000:{pcie:50,fp8:1,name:"RTX PRO 6000 96GB",sn:"RTX PRO 6000",v:"NVIDIA",tier:"ws",tf:500,bw:1.792,vram:96,tdp:600,rate:2.09,rsrc:RP,fleet:false,ftq:"no",mkt:{},maxN:8},
 rtx6000ada:{fp8:1,name:"RTX 6000 Ada 48GB",sn:"RTX 6000 Ada",v:"NVIDIA",tier:"ws",tf:364,bw:0.96,vram:48,tdp:300,rate:0.84,rsrc:RP,fleet:false,ftq:"x",mkt:{},maxN:8},
 rtx5000ada:{fp8:1,name:"RTX 5000 Ada 32GB",sn:"RTX 5000 Ada",v:"NVIDIA",tier:"ws",tf:262,bw:0.576,vram:32,tdp:250,rate:null,rsrc:"no rate on file",fleet:false,ftq:"x",mkt:{},maxN:8},
 rtxa6000:  {name:"RTX A6000 48GB",sn:"RTX A6000",v:"NVIDIA",tier:"ws",tf:154.8,bw:0.768,vram:48,tdp:300,rate:0.53,rsrc:RP,fleet:false,ftq:"x",mkt:{},maxN:8},
 rtxa5000:  {name:"RTX A5000 24GB",sn:"RTX A5000",v:"NVIDIA",tier:"ws",tf:111,bw:0.768,vram:24,tdp:230,rate:0.27,rsrc:RP,fleet:false,ftq:"x",mkt:{},maxN:8},
 rtx5090:   {pcie:50,fp8:1,name:"RTX 5090 32GB",sn:"RTX 5090",v:"NVIDIA",tier:"ws",tf:209.5,bw:1.792,vram:32,tdp:575,rate:0.99,rsrc:RP,fleet:false,ftq:"no",mkt:{},maxN:8},
 rtx5080:   {pcie:50,fp8:1,name:"RTX 5080 16GB",sn:"RTX 5080",v:"NVIDIA",tier:"ws",tf:112,bw:0.96,vram:16,tdp:360,rate:null,rsrc:"no rate on file",fleet:false,ftq:"no",mkt:{},maxN:8},
 rtx4090:   {fp8:1,name:"RTX 4090 24GB",sn:"RTX 4090",v:"NVIDIA",tier:"ws",tf:165,bw:1.008,vram:24,tdp:450,rate:0.74,rsrc:RP,fleet:false,ftq:"x",mkt:{},maxN:8},
 rtx4080:   {fp8:1,name:"RTX 4080 16GB",sn:"RTX 4080",v:"NVIDIA",tier:"ws",tf:97.5,bw:0.717,vram:16,tdp:320,rate:null,rsrc:"no rate on file",fleet:false,ftq:"x",mkt:{},maxN:8},
 rtx3090:   {name:"RTX 3090 24GB",sn:"RTX 3090",v:"NVIDIA",tier:"ws",tf:71,bw:0.936,vram:24,tdp:350,rate:0.5,rsrc:RP,fleet:false,ftq:"x",mkt:{},maxN:8},
 rtx4070ti: {fp8:1,name:"RTX 4070 Ti 12GB",sn:"RTX 4070 Ti",v:"NVIDIA",tier:"ws",tf:80,bw:0.504,vram:12,tdp:285,rate:null,rsrc:"no rate on file",fleet:false,ftq:"x",mkt:{},maxN:8},
 rtx3060:   {name:"RTX 3060 12GB",sn:"RTX 3060",v:"NVIDIA",tier:"ws",tf:25.6,bw:0.36,vram:12,tdp:170,rate:null,rsrc:"no rate on file",fleet:false,ftq:"x",mkt:{},maxN:8},
 // ---- AMD ----
 mi325x:    {nvbw:360,nvlink:1,fp8:1,name:"MI325X 256GB",sn:"MI325X",v:"AMD",tier:"dc",tf:1307,bw:6.0,vram:256,tdp:1000,rate:null,rsrc:"no rate on file",fleet:false,ftq:"no",mkt:{},maxN:8},
 mi300x:    {nvbw:360,nvlink:1,fp8:1,name:"MI300X 192GB",sn:"MI300X",v:"AMD",tier:"dc",tf:1307,bw:5.3,vram:192,tdp:750,rate:1.99,rsrc:VI,fleet:false,ftq:"no",mkt:{},maxN:8},
 mi250x:    {nvbw:160,nvlink:1,name:"MI250X 128GB",sn:"MI250X",v:"AMD",tier:"dc",tf:383,bw:3.2,vram:128,tdp:560,rate:null,rsrc:"no rate on file",fleet:false,ftq:"no",mkt:{},maxN:8},
 mi210:     {name:"MI210 64GB",sn:"MI210",v:"AMD",tier:"dc",tf:181,bw:1.6,vram:64,tdp:300,rate:null,rsrc:"no rate on file",fleet:false,ftq:"no",mkt:{},maxN:8},
 w7900:     {name:"Radeon PRO W7900 48GB",sn:"W7900",v:"AMD",tier:"ws",tf:122,bw:0.864,vram:48,tdp:295,rate:null,rsrc:"no rate on file",fleet:false,ftq:"no",mkt:{},maxN:8},
 rx7900xtx: {name:"RX 7900 XTX 24GB",sn:"RX 7900 XTX",v:"AMD",tier:"ws",tf:123,bw:0.96,vram:24,tdp:355,rate:null,rsrc:"no rate on file",fleet:false,ftq:"no",mkt:{},maxN:8},
 // ---- Apple (unified memory) ----
 m3ultra:   {name:"Mac Studio M3 Ultra 512GB",sn:"M3 Ultra",v:"Apple",tier:"mac",tf:28.4,bw:0.819,vram:512,tdp:180,rate:null,rsrc:"own hardware",fleet:false,ftq:"no",mkt:{},maxN:1,unified:1},
 m2ultra:   {name:"Mac Studio M2 Ultra 192GB",sn:"M2 Ultra",v:"Apple",tier:"mac",tf:27.2,bw:0.8,vram:192,tdp:120,rate:null,rsrc:"own hardware",fleet:false,ftq:"no",mkt:{},maxN:1,unified:1},
 m4max:     {name:"MacBook Pro M4 Max 128GB",sn:"M4 Max",v:"Apple",tier:"mac",tf:17,bw:0.546,vram:128,tdp:70,rate:null,rsrc:"own hardware",fleet:false,ftq:"no",mkt:{},maxN:1,unified:1},
 m3max:     {name:"MacBook Pro M3 Max 128GB",sn:"M3 Max",v:"Apple",tier:"mac",tf:14.2,bw:0.4,vram:128,tdp:78,rate:null,rsrc:"own hardware",fleet:false,ftq:"no",mkt:{},maxN:1,unified:1},
 m4pro:     {name:"Mac mini M4 Pro 64GB",sn:"M4 Pro",v:"Apple",tier:"mac",tf:8.5,bw:0.273,vram:64,tdp:45,rate:null,rsrc:"own hardware",fleet:false,ftq:"no",mkt:{},maxN:1,unified:1},
 // ---- Intel ----
 gaudi3:    {nvbw:480,nvlink:1,fp8:1,name:"Gaudi 3 128GB",sn:"Gaudi 3",v:"Intel",tier:"dc",tf:1835,bw:3.7,vram:128,tdp:900,rate:null,rsrc:"no rate on file",fleet:false,ftq:"no",mkt:{},maxN:8},
 gaudi2:    {nvbw:240,nvlink:1,fp8:1,name:"Gaudi 2 96GB",sn:"Gaudi 2",v:"Intel",tier:"dc",tf:432,bw:2.46,vram:96,tdp:600,rate:null,rsrc:"no rate on file",fleet:false,ftq:"no",mkt:{},maxN:8}
};
if(typeof module!=="undefined")module.exports={GPUS};else (typeof window!=="undefined"?window:globalThis).GPUS=GPUS;
