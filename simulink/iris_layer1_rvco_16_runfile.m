clear
load("python_data_rvco_iris_16.mat");

mclk_freq = 100e6;
modelname = "iris_layer1_rvco_16";
load_system(modelname);
set_param(modelname, 'MultithreadedSim', 'on');

idx = 1;
timesteps = 200;
t = (1:timesteps)/mclk_freq;
vin1 = zeros(timesteps, 2);
vin1(:, 1) = t(1, :);
vin1(:, 2) = input_spikes(idx, :, 1);
vin2 = zeros(timesteps, 2);
vin2(:, 1) = t(1, :);
vin2(:, 2) = input_spikes(idx, :, 2);
vin3 = zeros(timesteps, 2);
vin3(:, 1) = t(1, :);
vin3(:, 2) = input_spikes(idx, :, 3);
vin4 = zeros(timesteps, 2);
vin4(:, 1) = t(1, :);
vin4(:, 2) = input_spikes(idx, :, 4);

correct = 0;
for idx = 1:30
    vin1 = zeros(timesteps, 2);
    vin1(:, 1) = t(1, :);
    vin1(:, 2) = input_spikes(idx, :, 1);
    vin2 = zeros(timesteps, 2);
    vin2(:, 1) = t(1, :);
    vin2(:, 2) = input_spikes(idx, :, 2);
    vin3 = zeros(timesteps, 2);
    vin3(:, 1) = t(1, :);
    vin3(:, 2) = input_spikes(idx, :, 3);
    vin4 = zeros(timesteps, 2);
    vin4(:, 1) = t(1, :);
    vin4(:, 2) = input_spikes(idx, :, 4);

    tic
    out = sim(modelname, 1e-9 + (2 + timesteps)/mclk_freq);
    toc

    hidden_neurons = size(out.output_vector.Data, 1)/2;
    out_vector = out.output_vector.Data(1:hidden_neurons, 1, size(out.output_vector.Data, 3));
    spikes = out.output_vector.Data(hidden_neurons+1:2*hidden_neurons, 1, :);
    times = out.tout;
    spike_data = zeros(size(times, 1), hidden_neurons+1);
    spike_data(:, 1) = times(:, 1);
    for i = 1:hidden_neurons
        spike_data(:, i+1) = spikes(i, 1, :);
    end

    output = layer2_weights * out_vector + layer2_bias';
    [pred, class] = max(output);
    correct = correct + (class == (test_labels(s, idx)+1));

    print_string = sprintf("Simulation %d/30: Accuracy = %.4f.", idx, correct/idx);
    disp(print_string);
end

bdclose all
