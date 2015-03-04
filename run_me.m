% Assignment #4
% CS 536, Winter 2015
% Author: Michael Lam

clear;

%% Constants
DOCWORD_PATHS{1} = 'data/docword.nips.txt';
DOCWORD_PATHS{2} = 'data/docword.kos.txt';
VOCAB_PATHS{1} = 'data/vocab.nips.txt';
VOCAB_PATHS{2} = 'data/vocab.kos.txt';

%% Specify burn-in
BURN_IN_PERIOD = 100; % iterations

%% Run for each topic
for dataset = 1:2

    %% Read in data
    vocabWords = textread(VOCAB_PATHS{dataset}, '%s', 'delimiter', '\n');
    counts = dlmread(DOCWORD_PATHS{dataset}, ' ');
    nDocs = counts(1,1);
    nVocabWords = counts(2,1);
    nTotalWords = counts(3,1);
    counts = counts(4:end, :);
    
    %% Run for each topic
    for nTopics = [2 5 10]

        %% Priors (uniform)
        alpha = ones(nTopics, 1);
        beta = ones(nVocabWords, 1);
        
        %% Inference burn in period
        
        % set up initial counts and assignments
        topicAssignments = randi(nTopics, [nTotalWords 1]);
        N_dk = randi(nTotalWords, [nDocs nTopics]);
        N_kw = randi(nTotalWords, [nTopics nVocabWords]);
        N_k = randi(nTotalWords, [nTopics 1]);
        
        N_dk = floor((N_dk ./ sum(sum(N_dk)))*nTotalWords);
        N_kw = floor((N_kw ./ sum(sum(N_kw)))*nTotalWords);
        N_k = floor((N_k ./ sum(N_k))*nTotalWords);
        
        % iterate for burn-in period, then last iteration for sampling
        for iter = 1:BURN_IN_PERIOD+1
            fprintf('iter %d...', iter);
            tic;
            for i = 1:length(topicAssignments)
                % variables
                doc = counts(i, 1);
                word = counts(i, 2);
                cnts = counts(i, 3);
                topic = topicAssignments(i);
                
                % decrement counts: remove current assignment
                N_dk(doc, topic) = max(N_dk(doc, topic)-1, 0);
                N_kw(topic, word) = max(N_kw(topic, word)-1, 0);
                N_k(topic) = max(N_k(topic)-1, 0);
                
                % compute probability of each topic assignment
                P = zeros([nTopics 1]);
                for k = 1:nTopics
                    P(k) = (N_dk(doc, k) + alpha(k))...
                        * (N_kw(k,word) + beta(word))/(N_k(k) + beta(word) * nVocabWords);
                end
                P = P./sum(P);
                
                % sample from distribution above to assign topic
                topic = find(mnrnd(1, P'));
                topicAssignments(i) = topic;
                
                % update counts
                N_dk(doc, topic) = N_dk(doc, topic) + 1;
                N_kw(topic, word) = N_kw(topic, word) + 1;
                N_k(topic) = N_k(topic) + 1;
            end
            
            % save progress in case need to interrupt
            if mod(iter, 10) == 0
                save(sprintf('state_dataset%d_topics%d.mat', dataset, nTopics),...
                    'topicAssignments', 'N_dk', 'N_kw', 'N_k', 'iter');
            end
            
            fprintf(' (%fs)\n', toc);
        end

        %% After burn-in: compute topic-word distribution
        psi = zeros(nTopics, nVocabWords);
        for k = 1:nTopics
            for w = 1:nVocabWords
                psi(k, w) = (N_kw(k, w) + beta(w))...
                    / (N_k(k) + nVocabWords*beta(w));
            end
        end
        
        %% Save variables
        save(sprintf('final_dataset%d_topics%d.mat', dataset, nTopics), 'N_dk', 'N_kw', 'N_k', 'psi');
        
        %% Print out top 10 most likely words and probabilities from each topic
        for k = 1:nTopics
           psi_k = [psi(k, :); 1:nVocabWords];
           psi_k = fliplr(sortrows(psi_k', 1)');

           fprintf('(Dataset="%s") Topic %d/%d:\n', VOCAB_PATHS{dataset}, k, nTopics);
           for i = 1:10
               fprintf('\t%s %f\n', vocabWords{psi_k(2, i)}, psi_k(1, i));
           end
        end
        
        fprintf('\n==============\n\n');
    end

end