import './style.css';
import Stats from 'stats.js/src/Stats.js';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { Controller } from './controller';

class Main {

  canvas: HTMLCanvasElement;
  loadingText: HTMLElement;
  loadingBar: HTMLElement;
  stats: Stats;
  clock: THREE.Clock;
  controller: Controller;

  constructor() {

    this.loadingText = document.getElementById('loading-text');
    this.loadingBar = document.getElementById('loading-bar');
    this.canvas = document.createElement('canvas');
    const devicePixelRatio = window.devicePixelRatio || 1;
    this.canvas.width =  window.innerWidth * devicePixelRatio;
    this.canvas.height =  window.innerHeight * devicePixelRatio;
    // this.canvas.width = 1920;
    // this.canvas.height = Math.round(1920 * this.canvas.clientHeight / this.canvas.clientWidth);
    this.controller = new Controller(this.canvas);

  }

  loading(percentage: number) {

    this.loadingText.innerHTML = `${percentage}%`;
    this.loadingBar.style.width = `${percentage}%`;

  }

  start() {
    
    const render = () => {
      this.stats.begin();
      this.controller.update();
      this.controller.run();
      this.stats.end();
      requestAnimationFrame(render);
    }

    render();

    // this.controller.debug();

  }

  async startTimestamp() {
    this.controller.initTimeStamp();
    while(1) {
      this.stats.begin();
      this.controller.update();
      await this.controller.runTimestamp();
      await new Promise(requestAnimationFrame);
      this.stats.end();
    }
  }

  async init() {

    const camera = new THREE.PerspectiveCamera( 50, window.innerWidth / window.innerHeight, 0.1, 30 );
    camera.position.set( 2, 2, 0 );
    camera.lookAt( 0, 0, 0 );
    new OrbitControls(camera, this.canvas);

    let light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set( -10, 20, -10 );

    this.loading(0);
    await this.controller.initWebGPU();
    this.loading(20);
    await this.controller.loadData((p) => this.loading(p));
    this.loading(95);
    await this.controller.initScene(camera, light);
    this.afterInit();

  }

  afterInit() {

    this.stats = new Stats();
    this.stats.showPanel( 0 ); // 0: fps, 1: ms, 2: mb, 3+: custom
    document.body.appendChild( this.stats.dom );
    this.controller.showConfigUI();
    document.getElementById('loading-container').remove();
    document.body.appendChild(this.canvas);

  }

}

const main = new Main();
await main.init();
main.start();
// await main.startTimestamp();