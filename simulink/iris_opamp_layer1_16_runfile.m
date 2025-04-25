clear
load("iris_model_opamp_10.mat");
load("input_data_10.mat");

mclk_freq = 100e6;
modelname = "iris_opamp_layer1_16_10";
load_system(modelname);
set_param(modelname, 'MultithreadedSim', 'on');

s = 1;
idx = 1;
timesteps = 200;
t = (1:timesteps)/mclk_freq;
vin1 = zeros(timesteps, 2);
vin1(:, 1) = t(1, :);
vin1(:, 2) = input_spikes(s, idx, :, 1);
vin2 = zeros(timesteps, 2);
vin2(:, 1) = t(1, :);
vin2(:, 2) = input_spikes(s, idx, :, 2);
vin3 = zeros(timesteps, 2);
vin3(:, 1) = t(1, :);
vin3(:, 2) = input_spikes(s, idx, :, 3);
vin4 = zeros(timesteps, 2);
vin4(:, 1) = t(1, :);
vin4(:, 2) = input_spikes(s, idx, :, 4);

TS = 1/mclk_freq;
timepoints = 1e-9+TS/2:TS:(200*TS);
spike_data = zeros(100, 30, 16, 200);

for s = 1:100
    disp(s);
    correct = 0;
    for idx = 1:30
        vin1 = zeros(timesteps, 2);
        vin1(:, 1) = t(1, :);
        vin1(:, 2) = input_spikes(s, idx, :, 1) * 5;
        vin2 = zeros(timesteps, 2);
        vin2(:, 1) = t(1, :);
        vin2(:, 2) = input_spikes(s, idx, :, 2) * 5;
        vin3 = zeros(timesteps, 2);
        vin3(:, 1) = t(1, :);
        vin3(:, 2) = input_spikes(s, idx, :, 3) * 5;
        vin4 = zeros(timesteps, 2);
        vin4(:, 1) = t(1, :);
        vin4(:, 2) = input_spikes(s, idx, :, 4) * 5;

        tic
        out = sim(modelname, 1e-9 + (2 + timesteps)/mclk_freq);
        toc

        hidden_neurons = size(out.output_vector.Data, 1)/2;
        out_vector = out.output_vector.Data(1:hidden_neurons, 1, size(out.output_vector.Data, 3));
        spikes = out.output_vector.Data(hidden_neurons+1:2*hidden_neurons, 1, :);
        times = out.tout;
        for i = 1:hidden_neurons
            spike_data(s, idx, i, :) = interp1(times, squeeze(spikes(i, 1, :)), timepoints);
        end

        output = layer2_weights * out_vector + layer2_bias';
        [pred, class] = max(output);
        correct = correct + (class == (test_labels(s, idx)+1));

        print_string = sprintf("Simulation %d/30: Accuracy = %.4f.", idx, correct/idx);
        disp(print_string);
    end
end

bdclose all

save("spike_data_10.mat", "spike_data");
