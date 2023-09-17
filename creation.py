import dtlpy as dl
from adapters.grounded_sam_adapter import GroundedSam
from adapters.sam_adapter import SegmentAnythingAdapter


def create_grounded_sam(project: dl.Project):
    metadata = dl.Package.get_ml_metadata(cls=GroundedSam,
                                          default_configuration={
                                              'classes': ['person', 'dog'],
                                              'box_threshold': 0.2,  # 0.25,
                                              'text_threshold': 0.2,  # 0.25,
                                              'nms_threshold': 0.8
                                          }
                                          )
    modules = dl.PackageModule.from_entry_point(entry_point='adapters/grounded_sam_adapter.py')

    package = project.packages.push(package_name='grounded-sam',
                                    # src_path=os.getcwd(),
                                    is_global=True,
                                    package_type='ml',
                                    codebase=dl.GitCodebase(
                                        git_url='https://github.com/dataloop-ai-apps/grounded-sam-adapter',
                                        git_tag='v0.1.10'),
                                    modules=[modules],
                                    service_config={
                                        'runtime': dl.KubernetesRuntime(pod_type=dl.INSTANCE_CATALOG_REGULAR_M,
                                                                        runner_image='gcr.io/viewo-g/piper/agent/runner/cpu/grounded-sam:0.1.1',
                                                                        autoscaler=dl.KubernetesRabbitmqAutoscaler(
                                                                            min_replicas=0,
                                                                            max_replicas=1),
                                                                        preemptible=True,
                                                                        concurrency=1).to_json(),
                                        'executionTimeout': 10 * 3600,
                                        'initParams': {'model_entity': None}
                                    },
                                    requirements=list(),
                                    metadata=metadata)

    model = package.models.create(model_name='grounded-sam',
                                  description='Object detection with text prompts, segment using SAM',
                                  tags=['sam', 'groundedDino' 'groundedSAM'],
                                  dataset_id=None,
                                  status='trained',
                                  scope='public',
                                  # scope='project',
                                  configuration={
                                      'classes': ['person', 'dog'],
                                      'box_threshold': 0.2,  # 0.25,
                                      'text_threshold': 0.2,  # 0.25,
                                      'nms_threshold': 0.8
                                  },
                                  project_id=package.project.id,
                                  labels=list(),
                                  input_type='image',
                                  output_type=[dl.AnnotationType.BOX,
                                               dl.AnnotationType.SEGMENTATION]
                                  )
    print(f'model and package deployed. package id: {package.id}, model id: {model.id}')
    return model


def create_sam(project: dl.Project):
    metadata = dl.Package.get_ml_metadata(cls=SegmentAnythingAdapter,
                                          default_configuration={'stability_score_threshold': 0.85,
                                                                 'predicted_iou_threshold': 0.85,
                                                                 'output_type': 'polygon'},
                                          output_type=[dl.AnnotationType.SEGMENTATION,
                                                       dl.AnnotationType.BOX,
                                                       dl.AnnotationType.POLYGON],
                                          )
    modules = dl.PackageModule.from_entry_point(entry_point='adapters/sam_adapter.py')

    package = project.packages.push(package_name='sam-model-adapter',
                                    codebase=dl.GitCodebase(
                                        git_url='https://github.com/dataloop-ai-apps/grounded-sam-adapter',
                                        git_tag='v0.1.8'),
                                    is_global=True,
                                    package_type='ml',
                                    modules=[modules],
                                    service_config={
                                        'runtime': dl.KubernetesRuntime(pod_type=dl.INSTANCE_CATALOG_REGULAR_M,
                                                                        runner_image='gcr.io/viewo-g/piper/agent/runner/cpu/grounded-sam:0.1.1',
                                                                        autoscaler=dl.KubernetesRabbitmqAutoscaler(
                                                                            min_replicas=0,
                                                                            max_replicas=1),
                                                                        preemptible=False,
                                                                        concurrency=1).to_json(),
                                        'executionTimeout': 10 * 3600,
                                        'initParams': {'model_entity': None}
                                    },
                                    metadata=metadata)

    model = package.models.create(model_name='segment-anything',
                                  description='Segment Anything model dataset',
                                  tags=['sam', 'pretrained', 'segmentation'],
                                  dataset_id=None,
                                  status='trained',
                                  scope='public',
                                  configuration={
                                      'stability_score_threshold': 0.85,
                                      'predicted_iou_threshold': 0.85,
                                      'output_type': 'polygon',
                                      'checkpoint_filepath': "artifacts/sam_vit_b_01ec64.pth",
                                      'checkpoint_url': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                                      'model_type': "vit_b",
                                      'id_to_label_map': {}},
                                  project_id=package.project.id,
                                  labels=list()
                                  )
    print(f'model and package deployed. package id: {package.id}, model id: {model.id}')
    return model


def deploy():
    dl.setenv('prod')
    project_name = 'DataloopModels'
    project = dl.projects.get(project_name)
    # project = dl.projects.get(project_id='0ebbf673-17a7-469c-bcb2-f00fdaedfc8b')


def update():
    f = dl.Filters(resource='services', use_defaults=False)
    f.add(field='packageId', values=package.id)
    sss = dl.services.list(filters=f)
    for s in sss.all():
        # print(s.runtime.runner_image)
        # s.runtime.runner_image ='dataloop_runner-cpu/grounded-sam:0.1.1'
        s.package_revision = s.package.version
        s.update(True)
    f = dl.Filters(resource='models', use_defaults=False)
    f.add(field='packageId', values=package.id)
    sss = dl.models.list(filters=f)
    for s in sss.all():
        print(s.id)
        s.runtime.runner_image = 'dataloop_runner-cpu/grounded-sam:0.1.1'
        # s.package_revision = s.package.version
        s.update()
