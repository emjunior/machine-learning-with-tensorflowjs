async function plot(pointsArray, featureName) {
  tfvis.render.scatterplot(
    { name: `${featureName} vs House Price` },
    { values: [pointsArray], series: ["original"] },
    { xLabel: featureName, yLabel: "Price" }
  );
}

function normalize(tensor) {
  const min = tensor.min();
  const max = tensor.max();
  const normalizedTensor = tensor.sub(min).div(max.sub(min));
  return {
    tensor: normalizedTensor,
    min,
    max
  };
}

function denormalize(tensor, min, max) {
  const denormalizedTensor = tensor.mul(max.sub(min)).add(min);
  return denormalizedTensor;
}

function createModel() {
  const model = tf.sequential();

  model.add(tf.layers.dense({
    units: 1,
    useBias: true,
    activation: 'linear',
    inputDim: 1,
  }));

  const optimizer = tf.train.sgd(0.1);
  
  model.compile({
    loss: 'meanSquaredError',
    optimizer,

  })

  return model;
}

async function trainModel (model, trainingFeatureTensor, trainingLabelTensor) {

 const { onBatchEnd, onEpochEnd} = tfvis.show.fitCallbacks(
    {name: "Training Performance"},
    ['loss']

  )

  return model.fit(trainingFeatureTensor, trainingLabelTensor, {
    batchSize: 32,
    epochs: 20,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd,
    }
  });
}

async function run() {
  //Import data from CSV
    const houseSalesDataset = tf.data.csv(
    "http://127.0.0.1:5500/kc_house_data.csv"
  );
  

  //Extract x and y values to plot
  const pointsDataset = houseSalesDataset.map(record => ({
    x: record.sqft_living,
    y: record.price
  }));
  const points = await pointsDataset.toArray();
  if(points.length % 2 !== 0) {
      points.pop();
  }
  tf.util.shuffle(points);
  plot(points, "Square feet");

  //Extract Features (inputs)
  const featureValues = points.map(p => p.x);
  const featureTensor = tf.tensor2d(featureValues, [featureValues.length, 1]);

  //Extract Labels (outputs)
  const labelValues = points.map(p => p.y);
  const labelTensor = tf.tensor2d(labelValues, [labelValues.length, 1]);

  //normalize features and labels
  const normalizedFeature = normalize(featureTensor);
  const normalizedLabel = normalize(labelTensor);

  const [trainingFeatureTensor, testingFeatureTensor] =tf.split(normalizedFeature.tensor, 2);
  const [trainingLabelTensor, testingLabelTensor] =tf.split(normalizedLabel.tensor, 2);

  const model = createModel();
  tfvis.show.modelSummary({name: "Model summary"}, model);
  const layer = model.getLayer(undefined, 0);
  tfvis.show.layer({name: "Layer 1"}, layer);

  const result = await trainModel(model, trainingFeatureTensor, trainingLabelTensor);
  console.log(result);

  const trainingLoss = result.history.loss.pop();
  console.log(`Training set loss: ${trainingLoss}`);

  const validationLoss = result.history.val_loss.pop();
  console.log(`Validation set loss: ${validationLoss}`);

  const lossTensor = model.evaluate(testingFeatureTensor, testingLabelTensor);
  const loss = await lossTensor.dataSync();
  console.log(`Testing set loss: ${loss}`);

}

run();
