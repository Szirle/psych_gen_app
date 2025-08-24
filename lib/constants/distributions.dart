import 'dart:math' as math;

import 'package:psych_gen_app/model/manipulated_dimension_name.dart';

/// Generates a list of [numPoints] values representing a normal (Gaussian)
/// distribution over x in [0, 1], normalized so that the maximum value is 1.0.
List<double> generateNormalDistributionPoints({
  int numPoints = 100,
  double mean = 0.5,
  double stddev = 0.15,
}) {
  final values = List<double>.generate(numPoints, (i) {
    final x = numPoints == 1 ? 0.0 : i / (numPoints - 1);
    final exponent = -math.pow(x - mean, 2) / (2 * stddev * stddev);
    return math.exp(exponent);
  });

  // Normalize to [0, 1] by dividing by the maximum value
  final maxVal = values.fold<double>(0.0, (m, v) => v > m ? v : m);
  if (maxVal > 0) {
    for (var i = 0; i < values.length; i++) {
      values[i] = values[i] / maxVal;
    }
  }
  return values;
}

/// A map of normal distributions (100 points in [0, 1]) for each
/// [ManipulatedDimensionName]. For now, all dimensions use the same default
/// normal distribution centered at 0.5 with stddev 0.15.
final Map<ManipulatedDimensionName, List<double>> normalDistributions = {
  for (final name in ManipulatedDimensionName.values)
    name: generateNormalDistributionPoints(
        numPoints: 100, mean: 0.5, stddev: 0.15),
};
