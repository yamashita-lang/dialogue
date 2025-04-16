### code for the linear mixed-effects analysis

library(lmerTest) # version 3.1-3
file_path <- "/_path_to_csv_file_/data_for_lmerTest.csv"
data <- read.csv(file_path, header = TRUE)
head(data)

### Normalize data
data_exsubj <- data[, !(names(data) %in% c("subject"))]
z_data <- as.data.frame(lapply(data_exsubj, scale))
z_data$subject <- data$subject
head(z_data)

##############################################
### Separate Linguistic model (Fig. 2b)

## model 0, AIC = 1345.601
model_linear0 <- lmer(separate_model ~ layer*context_length + (1|subject), data=z_data)
step_linear0 <- step(model_linear0, ddf = "Kenward-Roger")
final_linear0 <- get_model(step_linear0)
AIC(final_linear0)

## model 1, AIC = 1129.453
model_linear <- lmer(separate_model ~ layer*context_length + (1+context_length|subject), data=z_data)
step_linear <- step(model_linear, ddf = "Kenward-Roger")
final_linear <- get_model(step_linear)
AIC(final_linear)

## model 2, AIC = 1085.944
model_quad0 <- lmer(separate_model ~ layer*context_length + I(layer^2) + I(context_length^2) + (1|subject), data=z_data)
step_quad0 <- step(model_quad0, ddf = "Kenward-Roger")
final_quad0 <- get_model(step_quad0)
AIC(final_quad0)

## model 3, AIC = 667.1813
model_quad <- lmer(separate_model ~ layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length|subject), data=z_data)
step_quad <- step(model_quad, ddf = "Kenward-Roger")
final_quad <- get_model(step_quad)
AIC(final_quad)

## model 4, AIC = 507.3323
model_quad2 <- lmer(separate_model ~ layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length+I(context_length^2)|subject), data=z_data)
step_quad2 <- step(model_quad2, ddf = "Kenward-Roger")
final_quad2 <- get_model(step_quad2)
AIC(final_quad2)

## model 5, AIC = 514.1572
model_quad2 <- lmer(separate_model ~ layer + layer:context_length + I(layer^2) + I(context_length^2) + (1+context_length+I(context_length^2)|subject), data=z_data)
step_quad2 <- step(model_quad2, ddf = "Kenward-Roger")
final_quad2 <- get_model(step_quad2)
AIC(final_quad2)


summary(final_quad2)
confint(final_quad2)



##############################################
### Cross-modality prediction (Fig. 2b)

## model 0, AIC = 941.6546
model_linear0 <- lmer(cross_modality ~ layer*context_length + (1|subject), data=z_data)
step_linear0 <- step(model_linear0, ddf = "Kenward-Roger")
final_linear0 <- get_model(step_linear0)
AIC(final_linear0)

## model 1, AIC = 501.8521
model_linear <- lmer(cross_modality ~ layer*context_length + (1+context_length|subject), data=z_data)
step_linear <- step(model_linear, ddf = "Kenward-Roger")
final_linear <- get_model(step_linear)
AIC(final_linear)

## model 2, AIC = 881.3272
model_quad0 <- lmer(cross_modality ~ layer*context_length + I(layer^2) + I(context_length^2) + (1|subject), data=z_data)
step_quad0 <- step(model_quad0, ddf = "Kenward-Roger")
final_quad0 <- get_model(step_quad0)
AIC(final_quad0)

## model 3, AIC = 368.5605
model_quad <- lmer(cross_modality ~ layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length|subject), data=z_data)
step_quad <- step(model_quad, ddf = "Kenward-Roger")
final_quad <- get_model(step_quad)
AIC(final_quad)

## model 4, AIC = 743.7006
model_quad2 <- lmer(cross_modality ~ layer*context_length + I(layer^2) + I(context_length^2) + (1+I(context_length^2)|subject), data=z_data)
step_quad2 <- step(model_quad2, ddf = "Kenward-Roger")
final_quad2 <- get_model(step_quad2)
AIC(final_quad2)

## model 5, AIC = -46.54442
model_quad3 <- lmer(cross_modality ~ layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length+I(context_length^2)|subject), data=z_data)
step_quad3 <- step(model_quad3, ddf = "Kenward-Roger")
final_quad3 <- get_model(step_quad3)
AIC(final_quad3)

summary(final_quad3)
confint(final_quad3)


##############################################
### Unified Linguistic model (Fig. 2b)

## model 0, AIC = 1376.107
model_linear0 <- lmer(unified_model ~ layer*context_length + (1|subject), data=z_data)
step_linear0 <- step(model_linear0, ddf = "Kenward-Roger")
final_linear0 <- get_model(step_linear0)
AIC(final_linear0)

## model 1, AIC = 1108.267
model_linear <- lmer(unified_model ~ layer*context_length + (1+context_length|subject), data=z_data)
step_linear <- step(model_linear, ddf = "Kenward-Roger")
final_linear <- get_model(step_linear)
AIC(final_linear)

## model 2, AIC = 1142.161
model_quad0 <- lmer(unified_model ~ layer*context_length + I(layer^2) + I(context_length^2) + (1|subject), data=z_data)
step_quad0 <- step(model_quad0, ddf = "Kenward-Roger")
final_quad0 <- get_model(step_quad0)
AIC(final_quad0)

## model 3, AIC = 654.2653
model_quad <- lmer(unified_model ~ layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length|subject), data=z_data)
step_quad <- step(model_quad, ddf = "Kenward-Roger")
final_quad <- get_model(step_quad)
AIC(final_quad)

## model 4, AIC = 330.6091
model_quad2 <- lmer(unified_model ~ layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length+I(context_length^2)|subject), data=z_data)
step_quad2 <- step(model_quad2, ddf = "Kenward-Roger")
final_quad2 <- get_model(step_quad2)
AIC(final_quad2)

## model 5, AIC = 329.1611
# integrated ~ layer + I(layer^2) + (1 + context_length + I(context_length^2) | subject) +      layer:context_length
model_quad2 <- lmer(unified_model ~ layer + layer:context_length + I(layer^2) + I(context_length^2) + (1+context_length+I(context_length^2)|subject), data=z_data)
step_quad2 <- step(model_quad2, ddf = "Kenward-Roger")
final_quad2 <- get_model(step_quad2)
AIC(final_quad2)

summary(final_quad2)
confint(final_quad2)


##############################################
### Variance explained by Production-only model (Fig. 3a)

## model 0, AIC = 1207.517
model_linear0 <- lmer(var_explained_prod ~ layer*context_length + (1|subject), data=z_data)
step_linear0 <- step(model_linear0, ddf = "Kenward-Roger")
final_linear0 <- get_model(step_linear0)
AIC(final_linear0)

## model 1, AIC = 857.2111
model_linear <- lmer(var_explained_prod ~ layer*context_length + (1+context_length|subject), data=z_data)
step_linear <- step(model_linear, ddf = "Kenward-Roger")
final_linear <- get_model(step_linear)
AIC(final_linear)

## model 2, AIC = 1190.698
model_quad0 <- lmer(var_explained_prod ~ layer*context_length + I(layer^2) + I(context_length^2) + (1|subject), data=z_data)
step_quad0 <- step(model_quad0, ddf = "Kenward-Roger")
final_quad0 <- get_model(step_quad0)
AIC(final_quad0)

## model 3, AIC = 812.9651
# prod ~ layer * context_length + I(layer^2) + I(context_length^2) + (1 + context_length | subject)
model_quad <- lmer(var_explained_prod ~ layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length|subject), data=z_data)
step_quad <- step(model_quad, ddf = "Kenward-Roger")
final_quad <- get_model(step_quad)
AIC(final_quad)


## model 4, AIC = 542.9885
# prod ~ layer + context_length + I(layer^2) + (1 + context_length + I(context_length^2) | subject) +      layer:context_length
model_quad2 <- lmer(var_explained_prod ~ layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length+I(context_length^2)|subject), data=z_data)
step_quad2 <- step(model_quad2, ddf = "Kenward-Roger")
final_quad2 <- get_model(step_quad2)
AIC(final_quad2)

summary(final_quad2)
confint(final_quad2)


##############################################
### Variance explained by Comprehension-only model (Fig. 3a)

## model 0, AIC = 1272.916
model_linear0 <- lmer(var_explained_comp ~ layer*context_length + (1|subject), data=z_data)
step_linear0 <- step(model_linear0, ddf = "Kenward-Roger")
final_linear0 <- get_model(step_linear0)
AIC(final_linear0)

## model 1, AIC = 1166.598
model_linear <- lmer(var_explained_comp ~ layer*context_length + (1+context_length|subject), data=z_data)
step_linear <- step(model_linear, ddf = "Kenward-Roger")
final_linear <- get_model(step_linear)
AIC(final_linear)

## model 2, AIC = 1078.839
model_quad0 <- lmer(var_explained_comp ~ layer*context_length + I(layer^2) + I(context_length^2) + (1|subject), data=z_data)
step_quad0 <- step(model_quad0, ddf = "Kenward-Roger")
final_quad0 <- get_model(step_quad0)
AIC(final_quad0)

## model 3, AIC = 912.1216
model_quad <- lmer(var_explained_comp ~ layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length|subject), data=z_data)
step_quad <- step(model_quad, ddf = "Kenward-Roger")
final_quad <- get_model(step_quad)
AIC(final_quad)

## model 4, AIC = 468.6035
model_quad2 <- lmer(var_explained_comp ~ layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length+I(context_length^2)|subject), data=z_data)
step_quad2 <- step(model_quad2, ddf = "Kenward-Roger")
final_quad2 <- get_model(step_quad2)
AIC(final_quad2)

summary(final_quad2)
confint(final_quad2)


##############################################
### Variance explained by the intersection of Production and Comprehension (Fig. 4a)

## model 0, AIC = 1385.476
model_linear0 <- lmer(var_explained_intersection ~ layer*context_length + (1|subject), data=z_data)
step_linear0 <- step(model_linear0, ddf = "Kenward-Roger")
final_linear0 <- get_model(step_linear0)
AIC(final_linear0)

## model 1, AIC = 1189.982
model_linear <- lmer(var_explained_intersection ~ layer*context_length + (1+context_length|subject), data=z_data)
step_linear <- step(model_linear, ddf = "Kenward-Roger")
final_linear <- get_model(step_linear)
AIC(final_linear)

## model 2, AIC = 1143.516
model_quad0 <- lmer(var_explained_intersection ~ layer*context_length + I(layer^2) + I(context_length^2) + (1|subject), data=z_data)
step_quad0 <- step(model_quad0, ddf = "Kenward-Roger")
final_quad0 <- get_model(step_quad0)
AIC(final_quad0)

## model 3, AIC = 796.4492
model_quad <- lmer(var_explained_intersection ~ layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length|subject), data=z_data)
step_quad <- step(model_quad, ddf = "Kenward-Roger")
final_quad <- get_model(step_quad)
AIC(final_quad)

## model 4, AIC = 632.875
model_quad2 <- lmer(var_explained_intersection ~ layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length+I(context_length^2)|subject), data=z_data)
step_quad2 <- step(model_quad2, ddf = "Kenward-Roger")
final_quad2 <- get_model(step_quad2)
AIC(final_quad2)

summary(final_quad2)
confint(final_quad2)


##############################################
### Weight correlation, linguistic voxels (Fig. 2d)

## model 0, AIC = 581.1939
model_linear0 <- lmer(weightcorr_linguistic_voxels ~ layer*context_length + (1|subject), data=z_data)
step_linear0 <- step(model_linear0, ddf = "Kenward-Roger")
final_linear0 <- get_model(step_linear0)
AIC(final_linear0)

## model 1, AIC = 71.98705
model_linear <- lmer(weightcorr_linguistic_voxels ~ layer*context_length + (1+context_length|subject), data=z_data)
step_linear <- step(model_linear, ddf = "Kenward-Roger")
final_linear <- get_model(step_linear)
AIC(final_linear)

## model 2, AIC = 574.8981
model_quad0 <- lmer(weightcorr_linguistic_voxels ~ layer*context_length + I(layer^2) + I(context_length^2) + (1|subject), data=z_data)
step_quad0 <- step(model_quad0, ddf = "Kenward-Roger")
final_quad0 <- get_model(step_quad0)
AIC(final_quad0)

## model 3, AIC = 33.06361
model_quad <- lmer(weightcorr_linguistic_voxels ~ layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length|subject), data=z_data)
step_quad <- step(model_quad, ddf = "Kenward-Roger")
final_quad <- get_model(step_quad)
AIC(final_quad)

## model 4, AIC = -87.3876
model_quad2 <- lmer(weightcorr_linguistic_voxels ~ layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length+I(context_length^2)|subject), data=z_data)
step_quad2 <- step(model_quad2, ddf = "Kenward-Roger")
final_quad2 <- get_model(step_quad2)
AIC(final_quad2)

summary(final_quad2)
confint(final_quad2)


##############################################
### Weight correlation, cross-modal voxels (Fig 2d)

## model 0, AIC = 1323.639
model_linear0 <- lmer(weightcorr_cross_voxels ~ layer*context_length + (1|subject), data=z_data)
step_linear0 <- step(model_linear0, ddf = "Kenward-Roger")
final_linear0 <- get_model(step_linear0)
AIC(final_linear0)

## model 1, AIC = 1196.681
model_linear <- lmer(weightcorr_cross_voxels ~ layer*context_length + (1+context_length|subject), data=z_data)
step_linear <- step(model_linear, ddf = "Kenward-Roger")
final_linear <- get_model(step_linear)
AIC(final_linear)

## model 2, AIC = 1048.603
model_quad0 <- lmer(weightcorr_cross_voxels ~ layer*context_length + I(layer^2) + I(context_length^2) + (1|subject), data=z_data)
step_quad0 <- step(model_quad0, ddf = "Kenward-Roger")
final_quad0 <- get_model(step_quad0)
AIC(final_quad0)

## model 3, AIC = 805.0633
model_quad <- lmer(weightcorr_cross_voxels ~ layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length|subject), data=z_data)
step_quad <- step(model_quad, ddf = "Kenward-Roger")
final_quad <- get_model(step_quad)
summary(final_quad)
AIC(final_quad)

## model 4, AIC = 752.1483
model_quad2 <- lmer(weightcorr_cross_voxels ~ layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length+I(context_length^2)|subject), data=z_data)
step_quad2 <- step(model_quad2, ddf = "Kenward-Roger")
final_quad2 <- get_model(step_quad2)
AIC(final_quad2)

summary(final_quad2)
confint(final_quad2)


##############################################
### Weight correlation, production voxels (Fig. 4d)

## model 0, AIC = 586.8601
model_linear0 <- lmer(weightcorr_prod_voxels ~ layer*context_length + (1|subject), data=z_data)
step_linear0 <- step(model_linear0, ddf = "Kenward-Roger")
final_linear0 <- get_model(step_linear0)
AIC(final_linear0)

## model 1, AIC = 127.2933
model_linear <- lmer(weightcorr_prod_voxels ~ layer*context_length + (1+context_length|subject), data=z_data)
step_linear <- step(model_linear, ddf = "Kenward-Roger")
final_linear <- get_model(step_linear)
AIC(final_linear)

## model 2, AIC = 580.7251
model_quad0 <- lmer(weightcorr_prod_voxels ~ layer*context_length + I(layer^2) + I(context_length^2) + (1|subject), data=z_data)
step_quad0 <- step(model_quad0, ddf = "Kenward-Roger")
final_quad0 <- get_model(step_quad0)
AIC(final_quad0)

## model 3, AIC = 106.7707
model_quad <- lmer(weightcorr_prod_voxels ~ layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length|subject), data=z_data)
step_quad <- step(model_quad, ddf = "Kenward-Roger")
final_quad <- get_model(step_quad)
AIC(final_quad)

## model 4, AIC = -73.97554
model_quad2 <- lmer(weightcorr_prod_voxels ~ layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length+I(context_length^2)|subject), data=z_data)
step_quad2 <- step(model_quad2, ddf = "Kenward-Roger")
final_quad2 <- get_model(step_quad2)
AIC(final_quad2)

summary(final_quad2)
confint(final_quad2)


##############################################
### Weight correlation, comprehension voxels (Fig. 4d)

## model 0, AIC = 609.7327
model_linear0 <- lmer(weightcorr_comp_voxels ~ layer*context_length + (1|subject), data=z_data)
step_linear0 <- step(model_linear0, ddf = "Kenward-Roger")
final_linear0 <- get_model(step_linear0)
AIC(final_linear0)

## model 1, AIC = 104.5247
model_linear <- lmer(weightcorr_comp_voxels ~ layer*context_length + (1+context_length|subject), data=z_data)
step_linear <- step(model_linear, ddf = "Kenward-Roger")
final_linear <- get_model(step_linear)
AIC(final_linear)

## model 2, AIC = 610.158
model_quad0 <- lmer(weightcorr_comp_voxels ~ layer*context_length + I(layer^2) + I(context_length^2) + (1|subject), data=z_data)
step_quad0 <- step(model_quad0, ddf = "Kenward-Roger")
final_quad0 <- get_model(step_quad0)
AIC(final_quad0)

## model 3, AIC = 83.22362
model_quad <- lmer(weightcorr_comp_voxels ~ layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length|subject), data=z_data)
step_quad <- step(model_quad, ddf = "Kenward-Roger")
final_quad <- get_model(step_quad)
AIC(final_quad)

## model 4, AIC = -43.59121
model_quad2 <- lmer(weightcorr_comp_voxels ~ layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length+I(context_length^2)|subject), data=z_data)
step_quad2 <- step(model_quad2, ddf = "Kenward-Roger")
final_quad2 <- get_model(step_quad2)
AIC(final_quad2)

summary(final_quad2)
confint(final_quad2)


##############################################
### Weight correlation, bimodal voxels (Fig. 4d)

## model 0, AIC = 618.9067
model_linear0 <- lmer(weightcorr_bimodal_voxels ~ layer*context_length + (1|subject), data=z_data)
step_linear0 <- step(model_linear0, ddf = "Kenward-Roger")
final_linear0 <- get_model(step_linear0)
AIC(final_linear0)

## model 1, AIC = 153.4605
model_linear <- lmer(weightcorr_bimodal_voxels ~ layer*context_length + (1+context_length|subject), data=z_data)
step_linear <- step(model_linear, ddf = "Kenward-Roger")
final_linear <- get_model(step_linear)
AIC(final_linear)

## model 2, AIC = 605.2989
model_quad0 <- lmer(weightcorr_bimodal_voxels ~ layer*context_length + I(layer^2) + I(context_length^2) + (1|subject), data=z_data)
step_quad0 <- step(model_quad0, ddf = "Kenward-Roger")
final_quad0 <- get_model(step_quad0)
AIC(final_quad0)

## model 3, AIC = 110.7706
model_quad <- lmer(weightcorr_bimodal_voxels ~ layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length|subject), data=z_data)
step_quad <- step(model_quad, ddf = "Kenward-Roger")
final_quad <- get_model(step_quad)
AIC(final_quad)

## model 4, AIC = 44.11143
model_quad2 <- lmer(weightcorr_bimodal_voxels ~ layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length+I(context_length^2)|subject), data=z_data)
step_quad2 <- step(model_quad2, ddf = "Kenward-Roger")
final_quad2 <- get_model(step_quad2)
AIC(final_quad2)

summary(final_quad2)
confint(final_quad2)


##############################################
### Weight correlation between Unified Linguistic model vs. Preoduction weights in Separate Linguistic model (Fig. 3c)

## model 0, AIC = 791.619
model_linear0 <- lmer(weightcorr_uni_vs_prod ~ layer*context_length + (1|subject), data=z_data)
step_linear0 <- step(model_linear0, ddf = "Kenward-Roger")
final_linear0 <- get_model(step_linear0)
AIC(final_linear0)

## model 1, AIC = 684.8531
model_linear <- lmer(weightcorr_uni_vs_prod ~ layer*context_length + (1+context_length|subject), data=z_data)
step_linear <- step(model_linear, ddf = "Kenward-Roger")
final_linear <- get_model(step_linear)
AIC(final_linear)

## model 2, AIC = 487.8771
model_quad0 <- lmer(weightcorr_uni_vs_prod ~ layer*context_length + I(layer^2) + I(context_length^2) + (1|subject), data=z_data)
step_quad0 <- step(model_quad0, ddf = "Kenward-Roger")
final_quad0 <- get_model(step_quad0)
AIC(final_quad0)

## model 3, AIC = 269.3299
model_quad <- lmer(weightcorr_uni_vs_prod ~ layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length|subject), data=z_data)
step_quad <- step(model_quad, ddf = "Kenward-Roger")
final_quad <- get_model(step_quad)
AIC(final_quad)

##################################################################################################################
## model 4, AIC = 177.7319
model_quad2 <- lmer(weightcorr_uni_vs_prod ~ layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length+I(context_length^2)|subject), data=z_data)
step_quad2 <- step(model_quad2, ddf = "Kenward-Roger")
final_quad2 <- get_model(step_quad2)
AIC(final_quad2)

summary(final_quad2)
confint(final_quad2)


##############################################
### Weight correlation between Unified Linguistic model vs. Comprehension weights in Separate Linguistic model (Fig. 3c)

## model 0, AIC = 958.5771
model_linear0 <- lmer(weightcorr_uni_vs_comp ~ layer*context_length + (1|subject), data=z_data)
step_linear0 <- step(model_linear0, ddf = "Kenward-Roger")
final_linear0 <- get_model(step_linear0)
AIC(final_linear0)

## model 1, AIC = 778.2893
model_linear <- lmer(weightcorr_uni_vs_comp ~ layer*context_length + (1+context_length|subject), data=z_data)
step_linear <- step(model_linear, ddf = "Kenward-Roger")
final_linear <- get_model(step_linear)
AIC(final_linear)

## model 2, AIC = 882.3956
model_quad0 <- lmer(weightcorr_uni_vs_comp ~ layer*context_length + I(layer^2) + I(context_length^2) + (1|subject), data=z_data)
step_quad0 <- step(model_quad0, ddf = "Kenward-Roger")
final_quad0 <- get_model(step_quad0)
AIC(final_quad0)

## model 3, AIC = 668.8296
model_quad <- lmer(weightcorr_uni_vs_comp ~ layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length|subject), data=z_data)
step_quad <- step(model_quad, ddf = "Kenward-Roger")
final_quad <- get_model(step_quad)
AIC(final_quad)

##################################################################################################################
## model 4, AIC = 591.2576
model_quad2 <- lmer(weightcorr_uni_vs_comp ~ layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length+I(context_length^2)|subject), data=z_data)
step_quad2 <- step(model_quad2, ddf = "Kenward-Roger")
final_quad2 <- get_model(step_quad2)
AIC(final_quad2)

summary(final_quad2)
confint(final_quad2)



###################################################################
############  Model comparisons  ##################################
###################################################################
library(tidyr) # version 1.3.1

##############################################
### Separate Linguistic model same-modality vs. cross-modality predictions (Fig. 2b)
long_data <- pivot_longer(data, cols=c(separate_model, cross_modality), names_to = "Type", values_to = "score")

## model 0, AIC = -7911.392
model_linear0 <- lmer(score ~ Type + layer*context_length + (1+Type|subject), data=long_data)
step_linear0 <- step(model_linear0, ddf = "Kenward-Roger")
final_linear0 <- get_model(step_linear0)
AIC(final_linear0)

## model 1, AIC = -8379.581
model_linear <- lmer(score ~ Type + layer*context_length + (1+context_length+Type|subject), data=long_data)
step_linear <- step(model_linear, ddf = "Kenward-Roger")
final_linear <- get_model(step_linear)
AIC(final_linear)

## model 2, AIC = -7909.853
model_quad0 <- lmer(score ~ Type + layer*context_length + I(layer^2) + I(context_length^2) + (1+Type|subject), data=long_data)
step_quad0 <- step(model_quad0, ddf = "Kenward-Roger")
final_quad0 <- get_model(step_quad0)
AIC(final_quad0)

## model 3, AIC = failed to converge
model_quad <- lmer(score ~ Type + layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length+Type|subject), data=long_data)
step_quad <- step(model_quad, ddf = "Kenward-Roger")
final_quad <- get_model(step_quad)

## model 4, AIC = failed to converge
model_quad2 <- lmer(score ~ Type + layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length+I(context_length^2)+Type|subject), data=long_data)
step_quad2 <- step(model_quad2, ddf = "Kenward-Roger")
final_quad2 <- get_model(step_quad2)

summary(final_linear)
confint(final_linear)


##############################################
### Separate Linguistic model vs. Unified Linguistic model predictions (Fig. 2b)
long_data <- pivot_longer(data, cols=c(separate_model, unified_model), names_to = "Type", values_to = "score")

## model 0, AIC = -8036.557
model_linear0 <- lmer(score ~ Type + layer*context_length + (1+Type|subject), data=long_data)
step_linear0 <- step(model_linear0, ddf = "Kenward-Roger")
final_linear0 <- get_model(step_linear0)
AIC(final_linear0)

## model 1, AIC = failed to converge
model_linear <- lmer(score ~ Type + layer*context_length + (1+context_length+Type|subject), data=long_data)
step_linear <- step(model_linear, ddf = "Kenward-Roger")
final_linear <- get_model(step_linear)

## model 2, AIC = -8501.448
model_quad0 <- lmer(score ~ Type + layer*context_length + I(layer^2) + I(context_length^2) + (1+Type|subject), data=long_data)
step_quad0 <- step(model_quad0, ddf = "Kenward-Roger")
final_quad0 <- get_model(step_quad0)
AIC(final_quad0)

## model 3, AIC = -8750.497
model_quad <- lmer(score ~ Type + layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length|subject), data=long_data)
step_quad <- step(model_quad, ddf = "Kenward-Roger")
final_quad <- get_model(step_quad)
AIC(final_quad)

## model 4, AIC = failed to converge
model_quad2 <- lmer(score ~ Type + layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length+I(context_length^2)|subject), data=long_data)
step_quad2 <- step(model_quad2, ddf = "Kenward-Roger")
final_quad2 <- get_model(step_quad2)

summary(final_quad)
confint(final_quad)


##############################################
### Weight correlation, Linguistic voxels vs. cross-modal voxels (Fig. 2d)
long_data <- pivot_longer(data, cols=c(weightcorr_linguistic_voxels, weightcorr_cross_voxels), names_to = "Type", values_to = "score")

## model 0, AIC = -4991.041
model_linear0 <- lmer(score ~ Type + layer*context_length + (1+Type|subject), data=long_data)
step_linear0 <- step(model_linear0, ddf = "Kenward-Roger")
final_linear0 <- get_model(step_linear0)
AIC(final_linear0)

## model 1, AIC = failed to converge
model_linear <- lmer(score ~ Type + layer*context_length + (1+context_length+Type|subject), data=long_data)
step_linear <- step(model_linear, ddf = "Kenward-Roger")
final_linear <- get_model(step_linear)

## model 2, AIC = -5081.696
model_quad0 <- lmer(score ~ Type + layer*context_length + I(layer^2) + I(context_length^2) + (1+Type|subject), data=long_data)
step_quad0 <- step(model_quad0, ddf = "Kenward-Roger")
final_quad0 <- get_model(step_quad0)
AIC(final_quad0)

## model 3, AIC = failed to converge
model_quad <- lmer(score ~ Type + layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length+Type|subject), data=long_data)
step_quad <- step(model_quad, ddf = "Kenward-Roger")
final_quad <- get_model(step_quad)

## model 4, AIC = failed to converge
model_quad2 <- lmer(score ~ Type + layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length+I(context_length^2)+Type|subject), data=long_data)
step_quad2 <- step(model_quad2, ddf = "Kenward-Roger")
final_quad2 <- get_model(step_quad2)

summary(final_quad0)
confint(final_quad0)


##############################################
### Weight correlation, production voxels vs. bimodal voxels (Fig. 4d)
long_data <- pivot_longer(data, cols=c(weightcorr_prod_voxels, weightcorr_bimodal_voxels), names_to = "Type", values_to = "score")

## model 0, AIC = -5450.323
model_linear0 <- lmer(score ~ Type*context_length + layer*context_length + (1+Type|subject), data=long_data)
step_linear0 <- step(model_linear0, ddf = "Kenward-Roger")
final_linear0 <- get_model(step_linear0)
AIC(final_linear0)

## model 1, AIC = -6389.63
model_linear <- lmer(score ~ Type*context_length + layer*context_length + (1+context_length+Type|subject), data=long_data)
step_linear <- step(model_linear, ddf = "Kenward-Roger")
final_linear <- get_model(step_linear)
AIC(final_linear)

## model 2, AIC = -5432.764
model_quad0 <- lmer(score ~ Type*context_length + layer*context_length + I(layer^2) + I(context_length^2) + (1+Type|subject), data=long_data)
step_quad0 <- step(model_quad0, ddf = "Kenward-Roger")
final_quad0 <- get_model(step_quad0)
AIC(final_quad0)

## model 3, AIC = -6351.924
model_quad <- lmer(score ~ Type + layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length+Type|subject), data=long_data)
step_quad <- step(model_quad, ddf = "Kenward-Roger")
final_quad <- get_model(step_quad)
AIC(final_quad)

## model 4, AIC = failed to converge
model_quad2 <- lmer(score ~ Type + layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length+I(context_length^2)|subject), data=long_data)
step_quad2 <- step(model_quad2, ddf = "Kenward-Roger")
final_quad2 <- get_model(step_quad2)

summary(final_linear)
confint(final_linear)
##################################################################################################################


##############################################
### Weight correlation, comprehension voxels vs. bimodal voxels (Fig. 4d)
long_data <- pivot_longer(data, cols=c(weightcorr_comp_voxels, weightcorr_bimodal_voxels), names_to = "Type", values_to = "score")

## model 0, AIC = -5314.223
model_linear0 <- lmer(score ~ Type*context_length + layer*context_length + (1|subject), data=long_data)
step_linear0 <- step(model_linear0, ddf = "Kenward-Roger")
final_linear0 <- get_model(step_linear0)
AIC(final_linear0)

## model 1, AIC = -6245.591
model_linear <- lmer(score ~ Type + layer*context_length + (1+context_length|subject), data=long_data)
step_linear <- step(model_linear, ddf = "Kenward-Roger")
final_linear <- get_model(step_linear)
AIC(final_linear)

## model 2, AIC = -5298.305
model_quad0 <- lmer(score ~ Type + layer*context_length + I(layer^2) + I(context_length^2) + (1|subject), data=long_data)
step_quad0 <- step(model_quad0, ddf = "Kenward-Roger")
final_quad0 <- get_model(step_quad0)
AIC(final_quad0)

## model 3, AIC = -6266.899
model_quad <- lmer(score ~ Type + layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length|subject), data=long_data)
step_quad <- step(model_quad, ddf = "Kenward-Roger")
final_quad <- get_model(step_quad)
AIC(final_quad)

## model 4, AIC = failed to converge
model_quad2 <- lmer(score ~ Type + layer*context_length + I(layer^2) + I(context_length^2) + (1+context_length+I(context_length^2)|subject), data=long_data)
step_quad2 <- step(model_quad2, ddf = "Kenward-Roger")
final_quad2 <- get_model(step_quad2)

summary(final_quad)
confint(final_quad)

