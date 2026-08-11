import { GUI } from 'lil-gui';

class Config {

  private gui: GUI;

  public scnenOptions = {
    scene: 0,
    skybox: 1,
    particles: true,
    'Simulation Start/Pause': function() { }
  }

  public simulationOptions = {
    iteration: 5,
    XSPH: 0.01,
    vorticity: 0.1,
    surfaceTension: 0.5,
    gravity: 9.8
  };

  // Screen space fluid rendering. The filter parameters are multiples of the
  // imposter radius, following the paper's delta = 10r / mu = r / sigma = 0.7r.
  public renderOptions = {
    renderMode: 0,          // 0: fluid surface, 1: raw particles
    particleScale: 2.0,     // imposter radius / simulation particle radius
    iteration: 2,           // narrow-range filter iterations (2 x 1D each)
    cleanUp: true,          // final 5x5 2D pass removing the 1D streaks
    filterSigma: 0.7,
    filterDelta: 10.0,
    filterMu: 1.0,
    maxFilterSigma: 12,     // upper bound of the screen space kernel size (pixel)
    ior: 1.333,
    absorption: 12.0,
    opacity: 40.0,
    fluidColor: '#3fa9f5'
  };

  constructor() {
    this.gui = new GUI();
    this.gui.hide();
  }

  public initSceneOptions(onChangeFunc: (msg) => void, switchFunc: () => void) {

    this.scnenOptions['Simulation Start/Pause'] = switchFunc;
    const sceneOptionGUI = this.gui.addFolder('Scene Options');
    sceneOptionGUI.add(this.scnenOptions, 'scene',
      { 'Bunny Drop': 0, 'Cube Drop': 1, 'Water Droplet': 2, 'Double Dam Break': 3, 'Boundary': 4 }
    );
    sceneOptionGUI.add(this.scnenOptions, 'skybox',
      { 'None': 0, 'Sea': 1 }
    );
    sceneOptionGUI.add(this.scnenOptions, 'particles');
    sceneOptionGUI.add(this.scnenOptions, 'Simulation Start/Pause');
    sceneOptionGUI.onFinishChange(onChangeFunc);

  }

  public initSimulationOptions(onChangeFunc: (msg) => void) {

    const simulationOptionGUI = this.gui.addFolder('Fluid Simulation Options');
    simulationOptionGUI.add(this.simulationOptions, 'iteration', 1, 10).step(5);
    simulationOptionGUI.add(this.simulationOptions, 'XSPH', 0, 0.1);
    simulationOptionGUI.add(this.simulationOptions, 'vorticity', 0, 1);
    simulationOptionGUI.add(this.simulationOptions, 'surfaceTension', 0, 1);
    simulationOptionGUI.add(this.simulationOptions, 'gravity', 0, 10);
    simulationOptionGUI.onFinishChange(onChangeFunc);
    
  }

  public initRenderOptions(onChangeFunc: (msg) => void) {

    const renderOptionGUI = this.gui.addFolder('Fluid Render Options');
    renderOptionGUI.add(this.renderOptions, 'renderMode',
      { 'Fluid Surface': 0, 'Raw Particles': 1 }
    );
    renderOptionGUI.add(this.renderOptions, 'particleScale', 1, 4);
    renderOptionGUI.add(this.renderOptions, 'iteration', 0, 4, 1);
    renderOptionGUI.add(this.renderOptions, 'cleanUp');
    renderOptionGUI.add(this.renderOptions, 'filterSigma', 0.2, 4);
    renderOptionGUI.add(this.renderOptions, 'filterDelta', 1, 30);
    renderOptionGUI.add(this.renderOptions, 'filterMu', 0.1, 10);
    renderOptionGUI.add(this.renderOptions, 'maxFilterSigma', 1, 16, 1);
    renderOptionGUI.add(this.renderOptions, 'ior', 1.0, 2.0);
    renderOptionGUI.add(this.renderOptions, 'absorption', 0, 60);
    renderOptionGUI.add(this.renderOptions, 'opacity', 1, 200);
    renderOptionGUI.addColor(this.renderOptions, 'fluidColor');
    renderOptionGUI.onFinishChange(onChangeFunc);

  }

  public show() {
    this.gui.show();
  }

}

export { Config };
