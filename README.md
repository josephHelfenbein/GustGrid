<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">

<a href="https://github.com/josephHelfenbein/recapgrid">
    <img src="src/textures/gustgrid.svg" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">GustGrid</h3>

  <p align="center">
    A CUDA-powered C++/OpenGL tool for real-time PC airflow and thermal simulation.
    <br />
    <br />
    <a href="https://github.com/josephHelfenbein/GustGrid/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/josephHelfenbein/GustGrid/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#prerequisites">Prerequisites</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

GustGrid is a high-performance simulation engine built in C++ and OpenGL, leveraging NVIDIA CUDA to deliver real-time visualization of PC airflow and thermal dynamics. By harnessing GPU-accelerated fluid dynamics, it accurately models heat dissipation, fan performance, and airflow patterns within complex computer chassis geometries. Its intuitive graphical interface allows users to interactively adjust component layouts and cooling configurations, enabling rapid design iterations and optimized thermal management for any PC build.


<b>Features:</b>
<ul>
<li><b>Voxel-based Fluid & Thermal Simulation:</b> Simulates a 64×256×128 voxel grid (volume pixels), with one CUDA thread per voxel for maximum parallelism.</li>

<li><b>All-in-one Physics:</b> Handles semi-Lagrangian advection, fan thrust, buoyancy, wall interactions, pressure projection, dissipation, convection, and conduction all on the GPU.</li>

<li><b>GPU-powered Pressure Solve:</b> Computes divergence, performs Jacobi iterations in shared memory, subtracts pressure gradients, and enforces boundary conditions to ensure incompressibility.</li>

<li><b>Advanced Heat Transfer:</b> Models convective (wind-chill style) and conductive (solid/fluid-specific diffusivity) heat exchange, alongside explicit heat sources and neighbor dissipation.</li>

<li><b>Interactive OpenGL Renderer:</b> Real-time ray marching volume heatmaps, PBR shading for chassis components, and custom controls for dynamic scene adjustments.</li>
</ul>


<img src="src/textures/screenshot1.png">
<img src="src/textures/screenshot2.png">
<img src="src/textures/screenshot3.png">

### Built With

* [![C++][C++]][c++-url]
* [![OpenGL][OpenGL]][OpenGL-url]
* [![CUDA][CUDA]][CUDA-url]



See the [open issues](https://github.com/josephHelfenbein/GustGrid/issues) for a full list of proposed features (and known issues).


### Physics Pipeline (per frame and voxel)

<ol>
<li><b>Update Fan Visibility:</b></li>
<ul><li>Ray march from each fan’s world position to the voxel center. If any solid voxel blocks the ray, fan influence is disabled for that fan.</li></ul>

<li><b>Velocity Update (advectKernel):</b>
<ul>
<li><b>Advection:</b> Backward-trace voxel positions along velocity, trilinearly sample previous velocity field.</li>

<li><b>Wall Proximity:</b> Scale advection strength by local solid proximity to simulate drag near chassis.</li>

<li><b>Fan Thrust:</b> Add axial and radial forces for voxels within a fan’s beam, attenuated by distance and alignment.</li>

<li><b>Buoyancy:</b> Apply upward acceleration for hot voxels using α·ΔT·g, capped for stability.</li>

<li><b>Thermal Swirl & Back-pressure:</b> Inject swirl in high ΔT regions and push fluid away from solid neighbors for hot pockets.</li>
</ul>
<li><b>Pressure Projection</b></li>
<ul>
<li><b>Divergence:</b> Compute ∇·u using central differences, treating solids as zero-velocity.</li>

<li><b>Jacobi Solver:</b> Iterate: load local tile in shared memory, average neighbor pressures, subtract divergence·scale, blend with previous pressure (β-relaxation).</li>

<li><b>Velocity Correction:</b> Subtract ∇p from velocity, incorporate thermal-pressure tweaks at boundaries, and reflect velocities into solids.</li>
</ul>
<li><b>Temperature Advection & Dissipation</b></li>
<ul>
<li><b>Forward-trace Advection:</b> Advect temperature using updated velocity, accumulate into tempSum/weightSum for weighted average.</li>

<li><b>Dissipation:</b> Compute per-voxel dissipation based on ΔT and velocity magnitude, then scatter dissipated energy into neighbors via tempSumDiss.</li>

<li><b>Normalize & Combine:</b> Compute advected temperature = tempSum/weightSum (or fallback), apply keepFraction, then add neighbor-dissipated heat.</li>
</ul>
<li><b>Convective & Conductive Heat Exchange</b></li>
<ul>
<li><b>Convective Heat:</b> For |ΔT|>threshold, loop over neighbors, compute heat transfer coefficients based on relative velocity and flow alignment, clamp and apply ∂T.</li>

<li><b>Conductive Diffusion:</b> Apply finite-difference diffusion with solid vs fluid diffusivity multipliers, add explicit heat sources, and clamp T to [ambient−10, ambient+200].</li>
</ol>

## Prerequisites

### Linux

To compile the project on Linux, you'll need:
1. **G++** - Install the `g++` compiler, version supporting at least C++ 17.
2. A CUDA-compatible NVIDIA GPU
3. **Libraries** - Use your package manager to install:
    - `libfreetype6-dev` (for font rendering)
    - `libglfw3-dev` (for creating windows, OpenGL contexts)
    - `cmake` (for building)
    - `cuda-tools` (for CUDA)
4. Run the `Linux Release (CMake)` configuration in VSCode, or `Linux Debug (CMake)` for debugging without optimization flags.



### Windows

To compile the project on Windows, you'll need:
1. **Visual Studio 2022** - Install Visual Studio 2022 at https://visualstudio.microsoft.com/downloads/ installing the `Linux, Mac, and embedded development with C++` toolset, and also the individual components `Windows 10 SDK` and `MSVC v143`.
2. **vcpkg** - Install vcpkg at https://learn.microsoft.com/en-us/vcpkg/get_started/get-started?pivots=shell-powershell and add environment variables
`VCPKG_ROOT=(PATH TO VCPKG)`
`PATH=(PATH TO VCPKG)`
4. Install needed packages with vcpkg
```bash
vcpkg install freetype glfw3
```
3. **CMake** - Install CMake at https://cmake.org/download/.
4. **CUDA** - Install CUDA at https://developer.nvidia.com/cuda-downloads and add environment variables
`CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\(YOUR CUDA VERSION)`
`PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\(YOUR CUDA VERSION)\bin`
`PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\(YOUR CUDA VERSION)\libnvvp`
5. Go to the project directory in `x64 Native Tools Command Prompt for VS 2022` and run `code .`, then run the `Windows Release (CMake)` configuration in VSCode, or `Windows Debug (CMake)` for debugging without optimization flags.
If you can't see any of the visuals, Windows may have overwritten the end of line sequences in the shaders to CRLF. You will need to update all of the shaders in src/shaders to have end of line sequence LF.


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Best README Template](https://github.com/othneildrew/Best-README-Template)
* [Learn OpenGL](https://learnopengl.com/)
* [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
* [Fast Fluid Dynamics Simulation on the GPU](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu)
* "(FREE) Gaming Pc" (https://skfb.ly/oGSTB) by Moonway 3D is licensed under Creative Commons Attribution (http://creativecommons.org/licenses/by/4.0/).
* "AM4 Cpu [ Free ]" (https://skfb.ly/pqx6R) by Igor.Jop is licensed under Creative Commons Attribution (http://creativecommons.org/licenses/by/4.0/).
* "Cooler Master CPU Cooler" (https://skfb.ly/oQV6T) by BlenderFace is licensed under Creative Commons Attribution (http://creativecommons.org/licenses/by/4.0/).
* "GeForce RTX 2060 Super Founders Edition" (https://skfb.ly/6Y7ww) by exéla is licensed under Creative Commons Attribution-NonCommercial (http://creativecommons.org/licenses/by-nc/4.0/).


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/josephHelfenbein/GustGrid.svg?style=for-the-badge
[contributors-url]: https://github.com/josephHelfenbein/GustGrid/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/josephHelfenbein/GustGrid.svg?style=for-the-badge
[forks-url]: https://github.com/josephHelfenbein/GustGrid/network/members
[stars-shield]: https://img.shields.io/github/stars/josephHelfenbein/GustGrid.svg?style=for-the-badge
[stars-url]: https://github.com/josephHelfenbein/GustGrid/stargazers
[issues-shield]: https://img.shields.io/github/issues/josephHelfenbein/GustGrid.svg?style=for-the-badge
[issues-url]: https://github.com/josephHelfenbein/GustGrid/issues
[license-shield]: https://img.shields.io/github/license/josephHelfenbein/GustGrid.svg?style=for-the-badge
[license-url]: https://github.com/josephHelfenbein/GustGrid/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/joseph-j-helfenbein
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[C++]: https://img.shields.io/badge/c++-00599C?logo=cplusplus&style=for-the-badge&logoColor=white
[c++-url]: https://developer.oracle.com/languages/javascript.html
[OpenGL]: https://img.shields.io/badge/opengl-5586A4?logo=opengl&style=for-the-badge&logoColor=white
[OpenGL-url]: https://www.khronos.org/webgl/
[CUDA]: https://img.shields.io/badge/cuda-76B900?logo=nvidia&style=for-the-badge&logoColor=white
[CUDA-url]: https://developer.nvidia.com/cuda-toolkit
