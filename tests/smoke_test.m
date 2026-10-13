function smoke_test
% Synthetic data checks wiring only; its accuracy is scientifically meaningless.
root = tempname; mkdir(root);
cleanup = onCleanup(@() rmdir(root,'s')); %#ok<NASGU>
rng(7);
for cls=1:2
    folder = fullfile(root,'data',sprintf('class%d',cls)); mkdir(folder);
    for i=1:20
        img = uint8(255*rand(32,32));
        imwrite(img,fullfile(folder,sprintf('%03d.png',i)));
    end
end
r = srnet_enas(fullfile(root,'data'),fullfile(root,'out'),'smoke');
assert(isfinite(r.accuracy) && sum(r.confusionMatrix(:))==8);
assert(isfile(fullfile(root,'out','candidate_log.csv')));
disp('Synthetic end-to-end smoke check passed.');
end
