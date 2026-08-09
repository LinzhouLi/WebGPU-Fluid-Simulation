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
      { 'None': 0, 'Sea': 1, 'Church': 2, 'Fall': 3, 'Mountain': 4 }
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

  public show() {
    this.gui.show();
  }

}

export { Config };
