// Our input frames will come from here.
const videoElement = document.getElementsByClassName("input_video")[0];
const canvasElement = document.getElementsByClassName("output_canvas")[0];
const controlsElement = document.getElementsByClassName("control-panel")[0];
const canvasCtx = canvasElement.getContext("2d");

// We'll add this to our control panel later, but we'll save it here so we can
// call tick() each time the graph runs.
const fpsControl = new FPS();

// Optimization: Turn off animated spinner after its hiding animation is done.
const spinner = document.querySelector(".loading");
spinner.ontransitionend = () => {
  spinner.style.display = "none";
};

function zColor(data) {
  return "white";
}

function onResults(results) {
  // Hide the spinner.
  document.body.classList.add("loaded");
  // Update the frame rate.
  fpsControl.tick();
  // Draw the overlays.
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(
    results.image,
    0,
    0,
    canvasElement.width,
    canvasElement.height
  );
  
  body_idxs = [11,12,13,14,15,16,23,24];
  var abdo_list = []; 
  for (var i=0; i<8; i++)
  {
    abdo_list.push(results.poseLandmarks[body_idxs[i]])
  }

  c_ab_x = (results.poseLandmarks[23].x + results.poseLandmarks[24].x) / 2;
  c_ab_y = (results.poseLandmarks[23].y + results.poseLandmarks[24].y) / 2;
  c_ab_z = (results.poseLandmarks[23].z + results.poseLandmarks[24].z) / 2;
  c_ab_v = (results.poseLandmarks[23].visibility + results.poseLandmarks[24].visibility) / 2;

  c_sh_x = (results.poseLandmarks[11].x + results.poseLandmarks[12].x) / 2;
  c_sh_y = (results.poseLandmarks[11].y + results.poseLandmarks[12].y) / 2;
  c_sh_z = (results.poseLandmarks[11].z + results.poseLandmarks[12].z) / 2;
  c_sh_v = (results.poseLandmarks[11].visibility + results.poseLandmarks[12].visibility) / 2;

  ratio_list = [1.5, 3, 6];
  abdo_list.push({'x': c_sh_x, 'y':c_sh_y, 'z': c_sh_z, 'visibility': c_sh_v});
  for (var i=0; i<3; i++)
  { 
    ratio = ratio_list[i];
    _x = (c_sh_x + (ratio - 1) * c_ab_x) / ratio;
    _y = (c_sh_y + (ratio - 1) * c_ab_y) / ratio;
    _z = (c_sh_z + (ratio - 1) * c_ab_z) / ratio;
    _v = (c_sh_v + (ratio - 1) * c_ab_v) / ratio;
    abdo_list.push({'x': _x, 'y':_y, 'z': _z, 'visibility': _v})
  }

  var head_idx = [7, 8]
  for (var i=0; i<2; i++)
  { 
    abdo_list.push(results.poseLandmarks[head_idx[i]])
  }

  for (var i=0; i<2; i++)
  {
    if(i==0)
    {
      l=1;
      r=4
    }
    else
    {
      l=10;
      r=9;
    }
    var _cx = results.poseLandmarks[l].x + results.poseLandmarks[r].x - results.poseLandmarks[0].x
    var _cy = results.poseLandmarks[l].y + results.poseLandmarks[r].y - results.poseLandmarks[0].y
    var _cz = results.poseLandmarks[l].z + results.poseLandmarks[r].z - results.poseLandmarks[0].z
    var _cv = results.poseLandmarks[l].visibility + results.poseLandmarks[r].visibility - results.poseLandmarks[0].visibility
    abdo_list.push({x:_cx, y:_cy, z:_cz, visibility:_cv})
  }
  for (var i=0; i<11; i++)
  {
    abdo_list.push(results.poseLandmarks[i])
  }

  for (var i=0; i<2; i++)
  {
    if(i==0)
    {
      l=1;
      r=4
    }
    else
    {
      l=10;
      r=9;
    }
    var _cx = (results.poseLandmarks[l].x + results.poseLandmarks[r].x)/2
    var _cy = (results.poseLandmarks[l].y + results.poseLandmarks[r].y)/2
    var _cz = (results.poseLandmarks[l].z + results.poseLandmarks[r].z)/2
    var _cv = (results.poseLandmarks[l].visibility + results.poseLandmarks[r].visibility)/2
    abdo_list.push({x:_cx, y:_cy, z:_cz, visibility:_cv})
  }


  drawLandmarks(
    canvasCtx,
    abdo_list,
    { visibilityMin: 0.15, color: zColor, fillColor: "rgb(18,38,243)" }
  );
  abdo_list.push(results.poseLandmarks[0])
  
    drawConnectors(canvasCtx, abdo_list, POSE_N_CONNECTIONS, {
    visibilityMin: 0.35,
    color: "rgb(142,239,131)",
  });

  canvasCtx.restore();
}

const pose = new Pose({
  locateFile: (file) => {
    return `${file}`;
  },
});
pose.setOptions({
  selfieMode: true,
  modelComplexity: 1,
  smoothLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
});
pose.onResults(onResults);

/**
 * Instantiate a camera. We'll feed each frame we receive into the solution.
 */
const camera = new Camera(videoElement, {
  onFrame: async () => {
    await pose.send({ image: videoElement });
  },
  width: 720,
  height: 720,
});
camera.start();

// Present a control panel through which the user can manipulate the solution
// options.
new ControlPanel(controlsElement, {
  selfieMode: true,
  modelComplexity: 1,
  smoothLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
})
  .add([
    new StaticText({ title: "TestPose" }),
    fpsControl,
    new Toggle({ title: "Selfie Mode", field: "selfieMode" }),
    new Slider({
      title: "Model Complexity",
      field: "modelComplexity",
      discrete: ["Lite", "Full", "Heavy"],
    }),
    new Toggle({ title: "Smooth Landmarks", field: "smoothLandmarks" }),
    new Slider({
      title: "Min Detection Confidence",
      field: "minDetectionConfidence",
      range: [0, 1],
      step: 0.01,
    }),
    new Slider({
      title: "Min Tracking Confidence",
      field: "minTrackingConfidence",
      range: [0, 1],
      step: 0.01,
    }),
  ])
  .on((options) => {
    videoElement.classList.toggle("selfie", options.selfieMode);
    pose.setOptions(options);
  });
