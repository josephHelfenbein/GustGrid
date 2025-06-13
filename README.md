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

(WIP)

GustGrid is a high-performance simulation engine built in C++ and OpenGL, leveraging NVIDIA CUDA to deliver real-time visualization of PC airflow and thermal dynamics. By harnessing GPU-accelerated fluid dynamics, it accurately models heat dissipation, fan performance, and airflow patterns within complex computer chassis geometries. Its intuitive graphical interface allows users to interactively adjust component layouts and cooling configurations, enabling rapid design iterations and optimized thermal management for any PC build.

### Built With

* [![C++][C++]][c++-url]
* [![OpenGL][OpenGL]][OpenGL-url]
* [![CUDA][CUDA]][CUDA-url]



See the [open issues](https://github.com/josephHelfenbein/GustGrid/issues) for a full list of proposed features (and known issues).

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

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Best README Template](https://github.com/othneildrew/Best-README-Template)
* [Learn OpenGL](https://learnopengl.com/)
* [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
* "(FREE) Gaming Pc" (https://skfb.ly/oGSTB) by Moonway 3D is licensed under Creative Commons Attribution (http://creativecommons.org/licenses/by/4.0/).
* "AM4 Cpu [ Free ]" (https://skfb.ly/pqx6R) by Igor.Jop is licensed under Creative Commons Attribution (http://creativecommons.org/licenses/by/4.0/).
* "Cooler Master CPU Cooler" (https://skfb.ly/oQV6T) by BlenderFace is licensed under Creative Commons Attribution (http://creativecommons.org/licenses/by/4.0/).


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
