import './style.css';
import Stats from 'stats.js/src/Stats.js';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { Controller } from './controller';

class Main {

  canvas: HTMLCanvasElement;
  stats: Stats;
  clock: THREE.Clock;
  controller: Controller;

  constructor() {

    this.canvas = document.querySelector('canvas');
    const devicePixelRatio = window.devicePixelRatio || 1;
    this.canvas.width = this.canvas.clientWidth * devicePixelRatio;
    this.canvas.height = this.canvas.clientHeight * devicePixelRatio;
    this.controller = new Controller(this.canvas);

    this.stats = new Stats();
    this.stats.showPanel( 0 ); // 0: fps, 1: ms, 2: mb, 3+: custom
    document.body.appendChild( this.stats.dom );

  }

  start() {
    
    const render = () => {
      this.stats.begin();
      
      this.controller.update();
      this.controller.run();

      this.stats.end();
      // requestAnimationFrame(render);
    }

    render();

  }

  async init() {

    const camera = new THREE.PerspectiveCamera( 50, window.innerWidth / window.innerHeight, 0.1, 50 );
    camera.position.set( 2, 2, 0 );
    camera.lookAt( 0, 0, 0 );
    new OrbitControls(camera, this.canvas);

    let light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set( 0, 0, 10 );

    await this.controller.initWebGPU();
    await this.controller.initScene(camera, light);

  }

}

const main = new Main();
main.init().then(() => main.start());