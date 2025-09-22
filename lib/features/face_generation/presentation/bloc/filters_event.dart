part of 'filters_bloc.dart';

abstract class FiltersEvent extends Equatable {
  const FiltersEvent();
  @override
  List<Object?> get props => [];
}

class LoadDistributionsEvent extends FiltersEvent {
  final int numPoints;
  final List<ManipulatedDimensionName>? variables;
  const LoadDistributionsEvent({this.numPoints = 100, this.variables});
}

class UpdateFilterEvent extends FiltersEvent {
  final ManipulatedDimensionName dimension;
  final List<double>? range; // [start, end] or null to clear
  final List<ManipulatedDimensionName>? variables;
  const UpdateFilterEvent({
    required this.dimension,
    this.range,
    this.variables,
  });
  @override
  List<Object?> get props => [dimension, range, variables];
}
